"""通用"从代码取证"能力:对全部 allowlist 代码建一个符号/定义的 AST 索引。

不是再配一域:任意问题都能一次查到相关的类/函数/模块常量的**定义签名 + docstring + 出处**,
而不是拿关键词反复抠片段(实测那样会 9 次撞上步数上限也答不出)。查不到就**明说**在哪些文件里
没找到——这就是"知道自己不知道"。

先索引模块级 + 类级定义；当符号定义不足以解释问题时，自动补充 allowlist 内的相关源码窗口，
从而覆盖函数内部局部状态、布尔条件和赋值。纯读取、不 import、不执行。
"""
from __future__ import annotations

import ast
import re
from typing import Any, Dict, List

from ..code_knowledge import _ALLOWLIST_FILES, _ROOT, _read_allowlisted
from .reward_deriver import _ref, _seg

_PY_FILES = tuple(f for f in _ALLOWLIST_FILES if f.endswith(".py"))

# 常见中文概念 → 代码里可能的英文符号词,帮中文提问也能命中。小而克制,不求全。
_ALIASES = {
    "观测": ["obs", "observation"],
    "网络": ["net", "model", "policy", "mlp", "actor", "critic", "encoder", "perceiver"],
    "结构": ["cfg", "config", "model"],
    "地形": ["terrain"],
    "扫描": ["scan", "raycast", "height"],
    "奖励": ["reward"],
    "课程": ["curriculum"],
    "步态": ["gait"],
    "接触": ["contact"],
    "执行器": ["actuator", "motor"],
    "参考": ["reference", "amp"],
    "遥测": ["telemetry"],
}

_STOP = {"the", "and", "for", "什么", "多少", "怎么", "哪些", "哪个", "如何", "是的"}


def _tokens(query: str) -> List[str]:
    """从提问里抽出可用于查符号的词:ASCII 标识符 + 中文别名映射。"""
    toks: List[str] = []
    for m in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query or ""):
        t = m.lower()
        if t not in _STOP and t not in toks:
            toks.append(t)
    for zh, en in _ALIASES.items():
        if zh in (query or ""):
            for e in en:
                if e not in toks:
                    toks.append(e)
    return toks


def _signature(node: ast.AST) -> str:
    try:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return f"def {node.name}({ast.unparse(node.args)})"
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            return f"class {node.name}({bases})" if bases else f"class {node.name}"
    except Exception:  # noqa: BLE001
        pass
    return getattr(node, "name", "") or ""


def _defs_in(body: List[ast.AST], rel: str, text: str, container: str = "") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append({
                "name": node.name, "kind": "method" if container else ("class" if isinstance(node, ast.ClassDef) else "function"),
                "container": container, "signature": _signature(node),
                "doc": (ast.get_docstring(node) or "")[:300],
                "ref": _ref(text, node, rel, symbol=node.name),
            })
            if isinstance(node, ast.ClassDef):
                out.extend(_defs_in(node.body, rel, text, container=node.name))
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out.append({
                        "name": tgt.id, "kind": "attr" if container else "const", "container": container,
                        "signature": _seg(text, node)[:120], "doc": "",
                        "ref": _ref(text, node, rel, symbol=tgt.id),
                    })
    return out


_index_cache: Dict[str, Any] = {}


def _mtime_sig() -> float:
    return sum((_ROOT / f).stat().st_mtime if (_ROOT / f).is_file() else 0.0 for f in _PY_FILES)


def _index() -> Dict[str, List[Dict[str, Any]]]:
    sig = _mtime_sig()
    if _index_cache.get("sig") != sig:
        idx: Dict[str, List[Dict[str, Any]]] = {}
        for rel in _PY_FILES:
            text = _read_allowlisted(rel)
            if not text:
                continue
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for d in _defs_in(tree.body, rel, text):
                idx.setdefault(d["name"].lower(), []).append(d)
        _index_cache["idx"] = idx
        _index_cache["sig"] = sig
    return _index_cache["idx"]


def get_code_facts(query: str = "", max_defs: int = 14) -> Dict[str, Any]:
    """任意问题 → 相关符号的定义 + 出处;并明说哪些词在 allowlist 代码里没找到。"""
    idx = _index()
    tokens = _tokens(query)
    hits: List[Dict[str, Any]] = []
    seen = set()
    matched = set()
    for tok in tokens:
        for name, entries in idx.items():
            # 只允许查询词包含在真实符号名中；反向匹配会把任意长的未知词
            # 因为恰好包含某个短符号名而错误标记为已找到。
            if tok == name or (len(tok) >= 3 and tok in name):
                matched.add(tok)
                for e in entries:
                    k = (e["ref"].file, e["ref"].line_start, e["name"])
                    if k in seen:
                        continue
                    seen.add(k)
                    exact = 0 if tok == name else 1
                    hits.append({"_rank": (exact, len(name)), **{
                        "name": e["name"], "kind": e["kind"], "container": e["container"],
                        "signature": e["signature"], "doc": e["doc"],
                        "file": e["ref"].file, "line": e["ref"].line_start,
                    }})
    hits.sort(key=lambda h: h["_rank"])
    for h in hits:
        h.pop("_rank", None)
    not_found = [t for t in tokens if t not in matched]
    from ..code_knowledge import search_code_knowledge

    raw = search_code_knowledge(query=query, max_snippets=8)
    source_context: Dict[str, Any] = {
        "snippets": raw.get("snippets") or [],
        "terms_used": raw.get("terms_used") or [],
        "matched_signal_rows": raw.get("matched_signal_rows") or [],
        "files_considered": raw.get("files_considered") or [],
        "gaps": raw.get("gaps") or [],
        "limitations": raw.get("limitations") or [],
    }
    return {
        "kind": "code_facts",
        "query": query,
        "tokens": tokens,
        "source_context": source_context,
        "definitions": hits[:max_defs],
        "not_found_symbols": not_found,
        "complete": bool(hits or source_context.get("snippets")),
        "note": ("优先返回 AST 定义；定义不足时补充 allowlist 源码窗口。"
                 "not_found_symbols 表示没有匹配到定义名，不等于源码中没有相关内部逻辑。"),
        "permission_boundary": "只读:全部来自本地 allowlist 代码的 AST 定义索引,不扫描任意路径、不远程、不写。",
    }

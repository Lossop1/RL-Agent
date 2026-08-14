#!/usr/bin/env bash
set -uo pipefail

# 在同一策略、命令和时长下分解 tier3 DR，避免 full 失败后无法定位物理来源。
PYTHON_BIN="${PYTHON_BIN:-/root/taili_sim2sim_env/bin/python}"
TOOL="${TOOL:-/root/taili_mujoco_long_horizon.py}"
MODEL="${MODEL:-/root/taili_rl_sar_cpp_0801/src/rl_sar_zoo/taili_description/mjcf/scene.xml}"
POLICY="${POLICY:-/root/taili_rl_sar_cpp_0801/policy/taili/taili_current/policy.pt}"
DR_SPEC="${DR_SPEC:-/root/taili_mujoco_dr_contract.json}"
OUT_DIR="${OUT_DIR:-/root/sim2sim_validation/taili_dr_matrix}"
STAND_SECONDS="${STAND_SECONDS:-2}"
MOVE_SECONDS="${MOVE_SECONDS:-12}"
COMMAND="${COMMAND:-0.5 0 0}"

mkdir -p "$OUT_DIR"
: > "$OUT_DIR/matrix.log"

run_case() {
    local name="$1"
    local profile="$2"
    local factors="$3"
    local seed="$4"
    local log="$OUT_DIR/$name.log"
    local summary="$OUT_DIR/$name.json"
    local factor_args=()
    if [[ -n "$factors" ]]; then
        factor_args=(--dr-factors "$factors")
    fi

    echo "CASE_START name=$name profile=$profile factors=$factors seed=$seed" | tee -a "$OUT_DIR/matrix.log"
    "$PYTHON_BIN" "$TOOL" \
        --model "$MODEL" \
        --policy "$POLICY" \
        --dr-spec "$DR_SPEC" \
        --dr-level 3 \
        --dr-profile "$profile" \
        "${factor_args[@]}" \
        --seed "$seed" \
        --command $COMMAND \
        --stand-seconds "$STAND_SECONDS" \
        --move-seconds "$MOVE_SECONDS" \
        --log-interval 3 \
        --summary-json "$summary" \
        --require-stable > "$log" 2>&1
    local rc=$?
    grep -E '^DR_SCENARIO |^MODEL_DR |^PUSH |^SUMMARY ' "$log" | tee -a "$OUT_DIR/matrix.log"
    echo "CASE_END name=$name rc=$rc" | tee -a "$OUT_DIR/matrix.log"
}

run_case single_mechanics single mechanics 101
run_case single_contact single contact 102
run_case single_actuation single actuation 103
run_case single_sensing single sensing 104
run_case single_push single push 105
run_case compound_mechanics_actuation compound mechanics,actuation 201
run_case compound_contact_sensing compound contact,sensing 202
run_case compound_actuation_push compound actuation,push 203
run_case full_301 full "" 301
run_case full_302 full "" 302

echo "MATRIX_DONE out=$OUT_DIR" | tee -a "$OUT_DIR/matrix.log"

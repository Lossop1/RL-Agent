import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import URDFLoader from "urdf-loader";
import {
  type DiagnosticPlayback,
  type DiagnosticPlaybackFrame,
} from "./api";
import { formatBackendText } from "./i18n/format";
import { zhCN as t } from "./i18n/zh-CN";

type ViewerState = "loading" | "ready" | "empty" | "error";

type UrdfRobot = THREE.Object3D & {
  joints?: Record<string, { setJointValue?: (value: number) => void }>;
};

const FOOT_COLORS = [0x70a7ff, 0xff786e, 0x7bd88f, 0xc49aff, 0xf5c451, 0x66c7cc];

const PLAYBACK_RATES = [0.25, 0.5, 1, 2, 4];

export default function RobotViewer({
  playback,
  active = true,
}: {
  playback: DiagnosticPlayback | null;
  active?: boolean;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const robotGroupRef = useRef<THREE.Group | null>(null);
  const robotRef = useRef<UrdfRobot | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const feetRef = useRef<Record<string, THREE.Mesh>>({});
  const playbackRef = useRef<DiagnosticPlayback | null>(null);
  const caseAnchorsRef = useRef<Map<string, DiagnosticPlaybackFrame>>(new Map());
  const playingRef = useRef(true);
  const cameraTrackingRef = useRef(false);
  const activeRef = useRef(active);
  const playbackRateRef = useRef(1);
  const playheadRef = useRef(0);
  const lastClockRef = useRef(0);
  const [viewerState, setViewerState] = useState<ViewerState>("loading");
  const [viewerError, setViewerError] = useState("");
  const [playing, setPlaying] = useState(true);
  const [cameraTracking, setCameraTracking] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [frameIndex, setFrameIndex] = useState(0);

  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);

  useEffect(() => {
    cameraTrackingRef.current = cameraTracking;
  }, [cameraTracking]);

  useEffect(() => {
    activeRef.current = active;
  }, [active]);

  useEffect(() => {
    playbackRateRef.current = playbackRate;
  }, [playbackRate]);

  useEffect(() => {
    playbackRef.current = playback;
    const anchors = new Map<string, DiagnosticPlaybackFrame>();
    for (const frame of playback?.frames ?? []) {
      const key = playbackCaseKey(frame);
      if (!anchors.has(key)) anchors.set(key, frame);
    }
    caseAnchorsRef.current = anchors;
    playheadRef.current = 0;
    setFrameIndex(0);
    setPlaying(Boolean(playback?.available && playback.frames.length));
    setViewerError("");
    setViewerState(
      playback === null
        ? "loading"
        : playback.available && playback.frames.length
          ? "ready"
          : "empty",
    );
  }, [playback]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x151814);
    const camera = new THREE.PerspectiveCamera(34, 1, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(1.65, -1.5, 1.1);

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch (reason) {
      console.error("Diagnostic WebGL initialization failed", reason);
      setViewerError("当前浏览器无法创建 WebGL 渲染上下文，请重启浏览器或启用硬件加速。");
      setViewerState("error");
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.25;
    controls.maxDistance = 15;
    controls.target.set(0, 0, 0.35);
    controlsRef.current = controls;

    scene.add(new THREE.HemisphereLight(0xf4f1e5, 0x283128, 2.4));
    const key = new THREE.DirectionalLight(0xffffff, 3.2);
    key.position.set(2.5, 4, 2);
    key.castShadow = true;
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x8cc8ff, 1.2);
    rim.position.set(-3, 1.5, -2);
    scene.add(rim);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(30, 30),
      new THREE.MeshStandardMaterial({ color: 0x1c211b, roughness: 0.95 })
    );
    ground.receiveShadow = true;
    scene.add(ground);
    const grid = new THREE.GridHelper(30, 120, 0x485246, 0x2c322b);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);
    // TERRAIN RECONSTRUCTION: thin contact outlines from foot-contact points.
    const terrainGroup = new THREE.Group();
    scene.add(terrainGroup);
    const terrainOutlines = new Map<string, THREE.LineSegments>();
    const terrainSolids = new Map<string, THREE.Mesh>();
    const outlineGeo = new THREE.EdgesGeometry(new THREE.PlaneGeometry(0.105, 0.105));
    let lastCaseKey = "";
    let lastFrameT = -1;
    let lastUiUpdateMs = -Infinity;

    const updateTerrainGeometry = (
      frame: DiagnosticPlaybackFrame | undefined,
      data: DiagnosticPlayback | null,
      anchorFrame: DiagnosticPlaybackFrame | undefined,
    ) => {
      const primitives = data?.scene?.terrain_primitives ?? [];
      const anchor = anchorFrame?.base_position ?? [0, 0, 0];
      const anchorTerrain = anchorFrame?.terrain_height ?? 0;
      ground.visible = primitives.length === 0;
      const primitiveKey = (primitive: typeof primitives[number]) =>
        `${data?.scene?.id || "scene"}:${primitive.id}:${primitive.type}:${primitive.size.join(",")}`;
      const known = new Set(primitives.map(primitiveKey));
      for (const [id, mesh] of terrainSolids) {
        if (!known.has(id)) {
          terrainGroup.remove(mesh);
          mesh.geometry.dispose();
          (mesh.material as THREE.Material).dispose();
          terrainSolids.delete(id);
        }
      }
      for (const primitive of primitives) {
        if (primitive.type !== "box" && primitive.type !== "plane") continue;
        const id = primitiveKey(primitive);
        let mesh = terrainSolids.get(id);
        if (!mesh) {
          const color = primitive.color || (primitive.type === "plane" ? "#30382f" : "#5f6d5e");
          const renderHeight = primitive.type === "plane" ? 0.01 : Math.max(0.01, primitive.size[2]);
          mesh = new THREE.Mesh(
            new THREE.BoxGeometry(
              Math.max(0.01, primitive.size[0]),
              Math.max(0.01, primitive.size[1]),
              renderHeight,
            ),
            new THREE.MeshStandardMaterial({ color, roughness: 0.9, metalness: 0.02 }),
          );
          mesh.receiveShadow = true;
          mesh.castShadow = primitive.type === "box";
          terrainSolids.set(id, mesh);
          terrainGroup.add(mesh);
        }
        mesh.position.set(
          primitive.center[0] - anchor[0],
          primitive.center[1] - anchor[1],
          primitive.center[2] - anchorTerrain - (primitive.type === "plane" ? 0.005 : 0),
        );
      }
      void frame;
    };

    const world = new THREE.Group();
    scene.add(world);
    robotGroupRef.current = world;

    const footMarkers: Record<string, THREE.Mesh> = {};
    const legOrder = playback?.robot?.leg_order ?? [];
    legOrder.forEach((leg, legIndex) => {
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.035, 16, 12),
        new THREE.MeshStandardMaterial({
          color: FOOT_COLORS[legIndex % FOOT_COLORS.length],
          emissive: FOOT_COLORS[legIndex % FOOT_COLORS.length],
          emissiveIntensity: 0.18,
          roughness: 0.45,
        })
      );
      marker.castShadow = true;
      footMarkers[leg] = marker;
      scene.add(marker);
    });
    feetRef.current = footMarkers;

    let disposed = false;
    const loader = new URDFLoader();
    const urdfUrl = playback?.robot?.urdf_url || playbackRef.current?.robot?.urdf_url || "";
    if (urdfUrl) loader.load(
      urdfUrl,
      (robot) => {
        if (disposed) return;
        const urdf = robot as UrdfRobot;
        urdf.updateMatrixWorld(true);
        urdf.traverse((object) => {
          if (!(object instanceof THREE.Mesh)) return;
          object.castShadow = true;
          object.receiveShadow = true;
          const name = object.parent?.name || object.name;
          const color = /foot/i.test(name)
            ? 0x383d37
            : /hip/i.test(name)
              ? 0x76b88a
              : /base/i.test(name)
                ? 0xe2e5dd
                : 0xaeb5aa;
          object.material = new THREE.MeshStandardMaterial({
            color,
            roughness: 0.62,
            metalness: 0.08,
          });
        });
        world.add(urdf);
        robotRef.current = urdf;
        applyPlaybackFrame(
          playbackRef.current?.frames[0],
          playbackRef.current,
          caseAnchorsRef.current,
          urdf,
          world,
          footMarkers,
        );
        setViewerState((current) => current === "empty" ? "empty" : "ready");
      },
      undefined,
      () => {
        setViewerError("机器人 URDF 资源加载失败，请刷新页面后重试。");
        setViewerState("error");
      }
    );
    else setViewerState("empty");

    const resize = () => {
      const width = Math.max(1, mount.clientWidth);
      const height = Math.max(1, mount.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);
    resize();

    let animation = 0;
    const render = (now: number) => {
      animation = requestAnimationFrame(render);
      if (!activeRef.current) {
        lastClockRef.current = now;
        return;
      }
      const data = playbackRef.current;
      const robot = robotRef.current;
      const worldGroup = robotGroupRef.current;
      if (playingRef.current && data?.available && data.frames.length && robot && worldGroup) {
        const last = data.frames[data.frames.length - 1]?.t ?? 0;
        const delta = lastClockRef.current ? Math.min(0.08, (now - lastClockRef.current) / 1000) : 0;
        playheadRef.current = last > 0 ? (playheadRef.current + delta * playbackRateRef.current) % last : 0;
        const index = frameIndexForTime(data.frames, playheadRef.current);
        const frame = data.frames[index];
        applyPlaybackFrame(frame, data, caseAnchorsRef.current, robot, worldGroup, footMarkers);
        if (now - lastUiUpdateMs >= 100) {
          lastUiUpdateMs = now;
          setFrameIndex((current) => current === index ? current : index);
        }
        let resetCaseView = false;
        if (frame) {
          // reset the reconstructed terrain when the playback wraps or switches case
          const currentCaseKey = playbackCaseKey(frame);
          resetCaseView = currentCaseKey !== lastCaseKey || frame.t < lastFrameT;
          if (resetCaseView) {
            terrainOutlines.forEach((m) => { terrainGroup.remove(m); (m.material as THREE.Material).dispose(); });
            terrainOutlines.clear();
            lastCaseKey = currentCaseKey;
          }
          lastFrameT = frame.t;
          const anchorFrame = caseAnchorsRef.current.get(currentCaseKey) ?? data.frames[0];
          const aPos = anchorFrame?.base_position ?? [0, 0, 0];
          const aTer = anchorFrame?.terrain_height ?? 0;
          updateTerrainGeometry(frame, data, anchorFrame);
          for (const foot of Object.values(frame.feet)) {
            if (!foot.contact) continue;
            const fx = foot.position[0] - aPos[0];
            const fy = foot.position[1] - aPos[1];
            const fz = foot.position[2] - aTer;
            const key = `${Math.round(fx / 0.1)},${Math.round(fy / 0.1)}`;
            let outline = terrainOutlines.get(key);
            if (!outline && terrainOutlines.size < 1500) {
              outline = new THREE.LineSegments(outlineGeo, new THREE.LineBasicMaterial({
                color: 0x8d948a,
                transparent: true,
                opacity: 0.24,
              }));
              terrainOutlines.set(key, outline);
              terrainGroup.add(outline);
            }
            if (outline) {
              outline.position.set(Math.round(fx / 0.1) * 0.1, Math.round(fy / 0.1) * 0.1, fz + 0.003);
              const material = outline.material as THREE.LineBasicMaterial;
              material.opacity = Math.max(0.14, Math.min(0.32, 0.18 + Math.abs(fz) * 0.08));
            }
          }
        }
        if (cameraTrackingRef.current) {
          // Preserve the user's orbit angle and zoom while following the robot.
          const dx = worldGroup.position.x - controls.target.x;
          const dy = worldGroup.position.y - controls.target.y;
          const dz = worldGroup.position.z - controls.target.z;
          if (Math.abs(dx) + Math.abs(dy) + Math.abs(dz) > 1e-5) {
            controls.target.x += dx; controls.target.y += dy; controls.target.z += dz * 0.2;
            camera.position.x += dx; camera.position.y += dy; camera.position.z += dz * 0.2;
          }
        }
        // snap the grid under the robot in cell-size steps so it reads as an infinite floor,
        // and FOLLOW the local ground height (descending terrain no longer puts the robot
        // visually "underground" beneath a fixed z=0 plane)
        const cell = 0.25;
        grid.position.x = Math.round(worldGroup.position.x / cell) * cell;
        grid.position.y = Math.round(worldGroup.position.y / cell) * cell;
        const anchorFrame = frame
          ? caseAnchorsRef.current.get(playbackCaseKey(frame)) ?? data.frames[0]
          : data.frames[0];
        const aTer0 = anchorFrame?.terrain_height ?? 0;
        const groundZ = (frame?.terrain_height ?? aTer0) - aTer0;
        grid.position.z = resetCaseView ? groundZ : grid.position.z + (groundZ - grid.position.z) * 0.25;
      }
      lastClockRef.current = now;
      controls.update();
      renderer.render(scene, camera);
    };
    render(0);

    return () => {
      disposed = true;
      cancelAnimationFrame(animation);
      observer.disconnect();
      controls.dispose();
      terrainOutlines.forEach((outline) => {
        (outline.material as THREE.Material).dispose();
      });
      terrainOutlines.clear();
      terrainSolids.forEach((mesh) => {
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
      });
      terrainSolids.clear();
      outlineGeo.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      scene.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return;
        object.geometry.dispose();
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        materials.forEach((material) => material.dispose());
      });
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
      robotGroupRef.current = null;
      robotRef.current = null;
      controlsRef.current = null;
      feetRef.current = {};
    };
  }, [playback?.robot?.urdf_url, playback?.robot?.leg_order.join(",")]);

  const currentFrame = playback?.frames[frameIndex] ?? null;
  const frameCount = playback?.frames.length ?? 0;
  const duration = useMemo(() => {
    if (!playback?.frames.length) return 0;
    return playback.frames[playback.frames.length - 1].t;
  }, [playback?.frames]);

  function resetView() {
    controlsRef.current?.reset();
  }

  function restartPlayback() {
    playheadRef.current = 0;
    setFrameIndex(0);
    setPlaying(true);
    applyPlaybackFrame(
      playback?.frames[0],
      playback,
      caseAnchorsRef.current,
      robotRef.current,
      robotGroupRef.current,
      feetRef.current,
    );
  }

  function showFrame(index: number, pause = true) {
    if (!playback?.frames.length) return;
    const clamped = Math.max(0, Math.min(playback.frames.length - 1, index));
    const frame = playback.frames[clamped];
    playheadRef.current = frame.t;
    setFrameIndex(clamped);
    if (pause) setPlaying(false);
    applyPlaybackFrame(
      frame,
      playback,
      caseAnchorsRef.current,
      robotRef.current,
      robotGroupRef.current,
      feetRef.current,
    );
  }

  const contactSummary = currentFrame
    ? Object.entries(currentFrame.feet)
        .filter(([, foot]) => foot.contact)
        .map(([leg]) => leg)
        .join(" ") || t.robotViewer.none
    : "--";

  return (
    <section className="robot-view">
      <div className="robot-canvas" ref={mountRef}>
        {viewerState === "loading" && <span className="robot-loading">{t.robotViewer.loading}</span>}
        {viewerState === "empty" && <span className="robot-loading">{t.robotViewer.empty}</span>}
        {viewerState === "error" && (
          <span className="robot-loading error">{viewerError || t.robotViewer.error}</span>
        )}
        <div className="robot-file-label">
          {playback?.available ? `${playback.source} playback / ${frameCount} frames` : t.robotViewer.record}
        </div>
      </div>
      <div className="robot-view-controls">
        <div>
          <p className="eyebrow">{t.robotViewer.eyebrow}</p>
          <h3>{t.robotViewer.title}</h3>
          <p>{formatBackendText(playback?.message) || t.robotViewer.fallbackMessage}</p>
        </div>
        <div className="pose-switch playback-controls">
          <button className={playing ? "active" : ""} onClick={() => setPlaying((value) => !value)} disabled={!frameCount}>
            {playing ? t.robotViewer.pause : t.robotViewer.play}
          </button>
          <button onClick={() => showFrame(frameIndex - 1)} disabled={!frameCount || frameIndex <= 0}>{t.robotViewer.prev}</button>
          <button onClick={() => showFrame(frameIndex + 1)} disabled={!frameCount || frameIndex >= frameCount - 1}>{t.robotViewer.next}</button>
          <button onClick={restartPlayback} disabled={!frameCount}>{t.robotViewer.restart}</button>
          <button onClick={resetView}>{t.robotViewer.resetView}</button>
          <button
            className={cameraTracking ? "active" : ""}
            aria-pressed={cameraTracking}
            title={cameraTracking ? t.robotViewer.cameraTrackingDisable : t.robotViewer.cameraTrackingEnable}
            onClick={() => setCameraTracking((value) => !value)}
          >
            {t.robotViewer.cameraTracking}
          </button>
        </div>
        <div className="playback-timeline">
          <input
            type="range"
            min={0}
            max={Math.max(0, frameCount - 1)}
            value={frameIndex}
            onChange={(event) => showFrame(Number(event.target.value))}
            disabled={!frameCount}
            aria-label={t.robotViewer.frameLabel}
          />
          <div className="playback-rates" aria-label={t.robotViewer.speedLabel}>
            {PLAYBACK_RATES.map((rate) => (
              <button
                className={playbackRate === rate ? "active" : ""}
                type="button"
                onClick={() => setPlaybackRate(rate)}
                key={rate}
              >
                {rate}x
              </button>
            ))}
          </div>
        </div>
        <div className="robot-observation">
          <span>{t.robotViewer.frame} <strong>{frameCount ? `${frameIndex + 1}/${frameCount}` : "--"}</strong></span>
          <span>{t.robotViewer.time} <strong>{currentFrame ? `${currentFrame.t.toFixed(2)} / ${duration.toFixed(2)} s` : "--"}</strong></span>
          <span>{t.robotViewer.terrain} <strong>{(currentFrame?.terrain || "--")}{currentFrame?.terrain_level != null ? `@L${currentFrame.terrain_level}` : ""}</strong></span>
          <span>{t.robotViewer.command} <strong>{currentFrame?.command_mode ?? "--"}</strong></span>
          <span>{t.robotViewer.stage} <strong>{currentFrame?.stage ?? "--"}</strong></span>
          <span>环境 <strong>{currentFrame ? currentFrame.env_id : playback?.selected_env_id ?? "--"}</strong></span>
          <span>{t.robotViewer.caseSegment} <strong>{currentFrame ? `${currentFrame.case_id} / ${currentFrame.segment_id}` : "--"}</strong></span>
          <span>{t.robotViewer.contacts} <strong>{contactSummary}</strong></span>
          <span>{t.robotViewer.rows} <strong>{playback ? `${playback.source_rows} raw, stride ${playback.stride}` : "--"}</strong></span>
        </div>
      </div>
    </section>
  );
}

function frameIndexForTime(frames: DiagnosticPlaybackFrame[], t: number) {
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (frames[mid].t <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

function applyPlaybackFrame(
  frame: DiagnosticPlaybackFrame | undefined,
  playback: DiagnosticPlayback | null | undefined,
  caseAnchors: Map<string, DiagnosticPlaybackFrame>,
  robot: UrdfRobot | null,
  world: THREE.Group | null,
  feet: Record<string, THREE.Mesh>
) {
  if (!frame || !robot || !world) return;
  const base = frame.base_position;
  // Each diagnostic case is spawned on a different world tile. Anchor within the case so
  // case transitions do not teleport the robot tens of metres out of the camera frustum.
  const anchorFrame = caseAnchors.get(playbackCaseKey(frame)) ?? playback?.frames[0];
  const anchor = anchorFrame?.base_position ?? [0, 0, 0];
  const anchorTerrain = anchorFrame?.terrain_height ?? 0;
  world.position.set(base[0] - anchor[0], base[1] - anchor[1], base[2] - anchorTerrain);
  const [w, x, y, z] = frame.base_quaternion_wxyz;
  world.quaternion.set(x, y, z, w).normalize();
  playback?.joint_order.forEach((jointName, index) => {
    const joint = robot.joints?.[jointName];
    const value = frame.joints[index];
    if (joint?.setJointValue && Number.isFinite(value)) joint.setJointValue(value);
  });
  for (const [leg, marker] of Object.entries(feet)) {
    const foot = frame.feet[leg];
    marker.visible = Boolean(foot);
    if (!foot) continue;
    marker.position.set(foot.position[0] - anchor[0], foot.position[1] - anchor[1], foot.position[2] - anchorTerrain);
    const scale = foot.contact ? 1.15 : 0.72;
    marker.scale.setScalar(scale);
    const material = marker.material as THREE.MeshStandardMaterial;
    material.opacity = foot.contact ? 1 : 0.45;
    material.transparent = !foot.contact;
  }
}

function playbackCaseKey(frame: DiagnosticPlaybackFrame) {
  return `${frame.stage}:${frame.case_id}:${frame.env_id}`;
}

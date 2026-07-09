import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import URDFLoader from "urdf-loader";
import {
  getDiagnosticPlayback,
  type DiagnosticPlayback,
  type DiagnosticPlaybackFrame,
  type DiagnosticReport,
} from "./api";
import { formatBackendText } from "./i18n/format";
import { zhCN as t } from "./i18n/zh-CN";

type ViewerState = "loading" | "ready" | "empty" | "error";

type UrdfRobot = THREE.Object3D & {
  joints?: Record<string, { setJointValue?: (value: number) => void }>;
};

const FOOT_COLORS: Record<string, number> = {
  FL: 0x70a7ff,
  FR: 0xff786e,
  RL: 0x7bd88f,
  RR: 0xc49aff,
};

const PLAYBACK_RATES = [0.25, 0.5, 1, 2, 4];

export default function RobotViewer({
  report,
  jobId,
  active = true,
}: {
  report: DiagnosticReport | null;
  jobId?: string | null;
  active?: boolean;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const robotGroupRef = useRef<THREE.Group | null>(null);
  const robotRef = useRef<UrdfRobot | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const feetRef = useRef<Record<string, THREE.Mesh>>({});
  const playbackRef = useRef<DiagnosticPlayback | null>(null);
  const playingRef = useRef(true);
  const activeRef = useRef(active);
  const playbackRateRef = useRef(1);
  const playheadRef = useRef(0);
  const lastClockRef = useRef(0);
  const [viewerState, setViewerState] = useState<ViewerState>("loading");
  const [playback, setPlayback] = useState<DiagnosticPlayback | null>(null);
  const [playing, setPlaying] = useState(true);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [frameIndex, setFrameIndex] = useState(0);

  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);

  useEffect(() => {
    activeRef.current = active;
  }, [active]);

  useEffect(() => {
    playbackRateRef.current = playbackRate;
  }, [playbackRate]);

  useEffect(() => {
    playbackRef.current = playback;
    playheadRef.current = 0;
    setFrameIndex(0);
    if (playback?.available && playback.frames.length) {
      setViewerState((current) => current === "loading" ? "loading" : "ready");
    }
  }, [playback?.output_dir, playback?.frames.length]);

  useEffect(() => {
    let cancelled = false;
    setViewerState("loading");
    getDiagnosticPlayback(900, jobId ?? report?.job_id)
      .then((next) => {
        if (cancelled) return;
        setPlayback(next);
        setViewerState(next.available && next.frames.length ? "ready" : "empty");
      })
      .catch(() => {
        if (!cancelled) setViewerState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [jobId, report?.job_id]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x151814);
    const camera = new THREE.PerspectiveCamera(34, 1, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(1.65, -1.5, 1.1);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.65;
    controls.maxDistance = 5;
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
      new THREE.PlaneGeometry(7, 7),
      new THREE.MeshStandardMaterial({ color: 0x1c211b, roughness: 0.95 })
    );
    ground.receiveShadow = true;
    scene.add(ground);
    const grid = new THREE.GridHelper(7, 28, 0x485246, 0x2c322b);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);
    // TERRAIN RECONSTRUCTION: thin contact outlines from foot-contact points.
    const terrainGroup = new THREE.Group();
    scene.add(terrainGroup);
    const terrainOutlines = new Map<string, THREE.LineSegments>();
    const outlineGeo = new THREE.EdgesGeometry(new THREE.PlaneGeometry(0.105, 0.105));
    let lastCaseId = -1;
    let lastFrameT = -1;

    const world = new THREE.Group();
    scene.add(world);
    robotGroupRef.current = world;

    const footMarkers: Record<string, THREE.Mesh> = {};
    for (const leg of ["FL", "FR", "RL", "RR"]) {
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.035, 16, 12),
        new THREE.MeshStandardMaterial({
          color: FOOT_COLORS[leg],
          emissive: FOOT_COLORS[leg],
          emissiveIntensity: 0.18,
          roughness: 0.45,
        })
      );
      marker.castShadow = true;
      footMarkers[leg] = marker;
      scene.add(marker);
    }
    feetRef.current = footMarkers;

    const loader = new URDFLoader();
    loader.load(
      "/robot/taili_dog_description/urdf/robot.urdf",
      (robot) => {
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
        applyPlaybackFrame(playbackRef.current?.frames[0], playbackRef.current, urdf, world, footMarkers);
        setViewerState((current) => current === "empty" ? "empty" : "ready");
      },
      undefined,
      () => setViewerState("error")
    );

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
        applyPlaybackFrame(frame, data, robot, worldGroup, footMarkers);
        setFrameIndex((current) => current === index ? current : index);
        if (frame) {
          // reset the reconstructed terrain when the playback wraps or switches case
          if (frame.case_id !== lastCaseId || frame.t < lastFrameT) {
            terrainOutlines.forEach((m) => { terrainGroup.remove(m); (m.material as THREE.Material).dispose(); });
            terrainOutlines.clear();
            lastCaseId = frame.case_id;
          }
          lastFrameT = frame.t;
          const aPos = data.frames[0]?.base_position ?? [0, 0, 0];
          const aTer = data.frames[0]?.terrain_height ?? 0;
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
        // FOLLOW CAMERA: keep the robot centered — shift orbit target AND camera by the same
        // delta so the user's orbit angle/zoom is preserved while the robot never leaves frame.
        const dx = worldGroup.position.x - controls.target.x;
        const dy = worldGroup.position.y - controls.target.y;
        const dz = worldGroup.position.z - controls.target.z;
        if (Math.abs(dx) + Math.abs(dy) + Math.abs(dz) > 1e-5) {
          controls.target.x += dx; controls.target.y += dy; controls.target.z += dz * 0.2;
          camera.position.x += dx; camera.position.y += dy; camera.position.z += dz * 0.2;
        }
        // snap the grid under the robot in cell-size steps so it reads as an infinite floor,
        // and FOLLOW the local ground height (descending terrain no longer puts the robot
        // visually "underground" beneath a fixed z=0 plane)
        const cell = 0.25;
        grid.position.x = Math.round(worldGroup.position.x / cell) * cell;
        grid.position.y = Math.round(worldGroup.position.y / cell) * cell;
        const aTer0 = data.frames[0]?.terrain_height ?? 0;
        const groundZ = (frame?.terrain_height ?? aTer0) - aTer0;
        grid.position.z += (groundZ - grid.position.z) * 0.25;
      }
      lastClockRef.current = now;
      controls.update();
      renderer.render(scene, camera);
    };
    render(0);

    return () => {
      cancelAnimationFrame(animation);
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      scene.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return;
        object.geometry.dispose();
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        materials.forEach((material) => material.dispose());
      });
      mount.removeChild(renderer.domElement);
      robotGroupRef.current = null;
      robotRef.current = null;
      controlsRef.current = null;
      feetRef.current = {};
    };
  }, []);

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
    applyPlaybackFrame(playback?.frames[0], playback, robotRef.current, robotGroupRef.current, feetRef.current);
  }

  function showFrame(index: number, pause = true) {
    if (!playback?.frames.length) return;
    const clamped = Math.max(0, Math.min(playback.frames.length - 1, index));
    const frame = playback.frames[clamped];
    playheadRef.current = frame.t;
    setFrameIndex(clamped);
    if (pause) setPlaying(false);
    applyPlaybackFrame(frame, playback, robotRef.current, robotGroupRef.current, feetRef.current);
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
        {viewerState === "error" && <span className="robot-loading error">{t.robotViewer.error}</span>}
        <div className="robot-file-label">
          {playback?.available ? `record.csv / ${frameCount} frames` : t.robotViewer.record}
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
          <span>Env <strong>{currentFrame ? currentFrame.env_id : playback?.selected_env_id ?? "--"}</strong></span>
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
  robot: UrdfRobot | null,
  world: THREE.Group | null,
  feet: Record<string, THREE.Mesh>
) {
  if (!frame || !robot || !world) return;
  const base = frame.base_position;
  // TRUE-WORLD z: subtract only the FIRST frame's ground height, so climbing (stairs/slope) renders
  // as real ascent instead of being flattened onto the grid every frame.
  const anchor = playback?.frames[0]?.base_position ?? [0, 0, 0];
  const anchorTerrain = playback?.frames[0]?.terrain_height ?? 0;
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

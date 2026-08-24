"""Robot-independent model-based locomotion control primitives.

The package deliberately has no dependency on a product, simulator, UI, or
remote runner. Product adapters supply a kinematics profile and a backend
converts the resulting joint command to a simulator or actuator interface.
"""

from .contracts import (
    BodyCommand,
    BodyTarget,
    ContactState,
    FootPlan,
    JointCommand,
    RobotState,
    WholeBodyDynamics,
)
from .controller import ControlStep, MpcWbcController
from .errors import ControlError, ControlSolveError
from .gait import GaitConfig, ContinuousTrotGait
from .mpc import CentroidalMpc, CentroidalMpcConfig, MpcOutput
from .servo import HighRateImpedanceServo
from .support import SupportGeometry, analyze_support_geometry
from .terrain import ContactTerrainEstimator, FlatTerrain, StairTerrain
from .wbc import (
    FloatingBaseWbc,
    WholeBodyConfig,
    WholeBodyDiagnostics,
    WholeBodySolution,
)

__all__ = [
    "BodyCommand",
    "BodyTarget",
    "CentroidalMpc",
    "CentroidalMpcConfig",
    "ContactState",
    "ContactTerrainEstimator",
    "ContinuousTrotGait",
    "ControlStep",
    "ControlError",
    "ControlSolveError",
    "FlatTerrain",
    "FootPlan",
    "FloatingBaseWbc",
    "GaitConfig",
    "HighRateImpedanceServo",
    "SupportGeometry",
    "analyze_support_geometry",
    "JointCommand",
    "MpcOutput",
    "MpcWbcController",
    "RobotState",
    "StairTerrain",
    "WholeBodyConfig",
    "WholeBodyDiagnostics",
    "WholeBodyDynamics",
    "WholeBodySolution",
]

from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution
from cgshop2026_pyutils.geometry import Point, FlippableTriangulation
from cgshop2026_pyutils.verify import check_for_errors

# Define points (square) and two triangulations that will be flipped to a common form
points_x = [0, 1, 0, 1]
points_y = [0, 0, 1, 1]
triangulations = [  # Each triangulation is a list of interior edges
	[(0, 3)],        # diagonal 0-3
	[(1, 2)],        # diagonal 1-2 (the flip partner)
]

instance = CGSHOP2026Instance(
	instance_uid="demo-square",
	points_x=points_x,
	points_y=points_y,
	triangulations=triangulations,
)

# A solution that flips the diagonal in the first triangulation to match the second.
# flips is: one list per triangulation -> sequence of parallel flip sets -> each set is a list of edges
solution = CGSHOP2026Solution(
	instance_uid="demo-square",
	flips=[ [[(0,3)]] , [] ]  # flip edge (0,3) in triangulation 0; triangulation 1 already in target form
)

errors = check_for_errors(instance, solution)
print("Errors:", errors or "None ✔")
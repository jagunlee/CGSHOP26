from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution
from cgshop2026_pyutils.verify import check_for_errors
import json

#solution=input("solution (/folder/instance.solution.json): ")
data=input("instance (instance.jason): ")

path = 'data/benchmark_instances/'
with open(path+data, 'r') as file:
    root = json.load(file)
    file.close()

instance = CGSHOP2026Instance(
        instance_uid=root["instance_uid"],
    points_x=root["points_x"],
    points_y=root["points_y"],
    triangulations=root["triangulations"],
)

print(f"# points = {len(root["points_x"])}")
print(f"# Ts = {len(root["triangulations"])}")
print(f"# edges = {len(root["triangulations"][0])}")

#with open(solution, 'r') as file:
#    root = json.load(file)
#    file.close()
#
#solution = CGSHOP2026Solution(
#        instance_uid=root["instance_uid"],
#        flips=root["flips"],
#        )
#
#errors = check_for_errors(instance, solution)
#print("Errors: ", errors or "None !")

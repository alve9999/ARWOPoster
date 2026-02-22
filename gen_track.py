import numpy as np
from scipy.spatial import Delaunay, ConvexHull

# --- Load cones ---
coords = []
colors = []
with open("cones.txt","r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts)!=3:
            continue
        color, x, y = parts
        coords.append([float(x), float(y)])
        colors.append(color)

points = np.array(coords)

# --- Delaunay triangulation ---
tri = Delaunay(points)

# --- Triangle filtering ---
max_edge_length = 6.5
min_angle_deg = 20

def edge_lengths(tri_pts):
    a = np.linalg.norm(tri_pts[0]-tri_pts[1])
    b = np.linalg.norm(tri_pts[1]-tri_pts[2])
    c = np.linalg.norm(tri_pts[2]-tri_pts[0])
    return np.array([a,b,c])

def triangle_angles(tri_pts):
    a,b,c = edge_lengths(tri_pts)
    angles = np.arccos([
        np.clip((b**2 + c**2 - a**2)/(2*b*c), -1,1),
        np.clip((a**2 + c**2 - b**2)/(2*a*c), -1,1),
        np.clip((a**2 + b**2 - c**2)/(2*a*b), -1,1)
    ])
    return np.degrees(angles)

filtered_simplices = []
for simplex in tri.simplices:
    tri_pts = points[simplex]
    if np.all(edge_lengths(tri_pts) <= max_edge_length) and np.min(triangle_angles(tri_pts)) >= min_angle_deg and np.mean(edge_lengths(tri_pts))>2.0:
        filtered_simplices.append(simplex)
filtered_simplices = np.array(filtered_simplices)

# --- Convex hull edges ---
hull = ConvexHull(points)
filtered_hull_edges = []
for i,j in hull.simplices:
    if np.linalg.norm(points[i]-points[j]) <= max_edge_length:
        filtered_hull_edges.append((i,j))

# --- Compute centerline along edges that cross the track (blue-yellow edges) ---
midpoints = []
# Use indices of blue and yellow cones in file order
for i, (color_i, pt_i) in enumerate(zip(colors, points)):
    if color_i != "BLUE":
        continue
    # look for nearest yellow cones among Delaunay neighbors
    neighbors = tri.vertex_neighbor_vertices[1][
        tri.vertex_neighbor_vertices[0][i] : tri.vertex_neighbor_vertices[0][i+1]
    ]
    for j in neighbors:
        if colors[j] == "YELLOW":
            xm, ym = (points[i]+points[j])/2
            midpoints.append([xm, ym])

midpoints = np.array(midpoints)# --- Generate TikZ ---
print(r"\begin{tikzpicture}[scale=0.3]")

# Triangles
for simplex in filtered_simplices:
    x0,y0 = points[simplex[0]]
    x1,y1 = points[simplex[1]]
    x2,y2 = points[simplex[2]]
    print(f"\\draw[red,line width=0.8pt] ({x0},{y0}) -- ({x1},{y1}) -- ({x2},{y2}) -- cycle;")

# Convex hull
for i,j in filtered_hull_edges:
    x0,y0 = points[i]
    x1,y1 = points[j]
    print(f"\\draw[red,line width=1.2pt] ({x0},{y0}) -- ({x1},{y1});")

# Cones
for (x,y), color in zip(points, colors):
    if(color=="BLUE"):
        print(f"\\fill[primary] ({x},{y}) circle (0.7);")
    elif(color=="YELLOW"):
        print(f"\\fill[secondary] ({x},{y}) circle (0.7);")

# Green centerline through midpoints
print("\\draw[green,line width=1.5pt] ", end="")
for idx,(x,y) in enumerate(midpoints[155:175]):
    if idx==0:
        print(f"({x},{y})", end=" ")
    else:
        print(f"-- ({x},{y})", end=" ")
print(";")
for x,y in midpoints[155:175]:
    print(f"\\node[black,circle,fill,inner sep=1.5pt] at ({x},{y}) {{}};")

print(r"\end{tikzpicture}")


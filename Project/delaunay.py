import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json, io

EPS = 10e-24
HISTORY = []

class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def distance(self, other):
    return ((self.x - other.x)**2 + (self.y - other.y)**2)

  def subtract(self, other):
    return Point(self.x - other.x, self.y - other.y)

  def __eq__(self, other):
    return self.distance(other) <= EPS

  def __hash__(self):
    return hash((round(self.x,5),round(self.x,5)))

  def __repr__(self):
    return f"({self.x}, {self.y})"


class Edge:
  def __init__(self, p1, p2):
    self.p1 = p1
    self.p2 = p2

  def __eq__(self, other):
    if not isinstance(other, Edge):
            return False
    return (self.p1 == other.p1 and self.p2 == other.p2 or self.p1==other.p2 and self.p2==other.p1)
  def __hash__(self):
    s = hash(self.p1)
    b = hash(self.p2)
    return hash((min(s,b),max(s,b)))

  def __repr__(self):
    return f"Edge({self.p1}, {self.p2})"


def orient(a,b,c):
    return (b.x-a.x)*(c.y-a.y)-(c.x-a.x)*(b.y-a.y)


class Triangle:
  def __init__(self, p1, p2, p3):
    if orient(p1,p2,p3) < 0:
      self.p1 = p1
      self.p2 = p3
      self.p3 = p2
    else:
      self.p1 = p1
      self.p2 = p2
      self.p3 = p3
    self.edges = [Edge(self.p1, self.p2), Edge(self.p2, self.p3), Edge(self.p3, self.p1)]
    self.neighbours = [None,None,None]
    self.is_bad = False
  def is_inside(self, p):
    return (    orient(self.p1, self.p2, p) >=0
            and orient(self.p2, self.p3, p) >=0
            and orient(self.p3, self.p1, p) >=0)
  def is_in_circumcircle(self, p):
    px, py = p.x, p.y
    p1x, p1y = self.p1.x, self.p1.y
    p2x, p2y = self.p2.x, self.p2.y
    p3x, p3y = self.p3.x, self.p3.y

    ax_ = p1x - px
    ay_ = p1y - py
    bx_ = p2x - px
    by_ = p2y - py
    cx_ = p3x - px
    cy_ = p3y - py

    det_a = ax_ * ax_ + ay_ * ay_
    det_b = bx_ * bx_ + by_ * by_
    det_c = cx_ * cx_ + cy_ * cy_

    det = (ax_ * (by_ * det_c - det_b * cy_) -
           ay_ * (bx_ * det_c - det_b * cx_) +
           det_a * (bx_ * cy_ - by_ * cx_))

    return det > EPS

  def __repr__(self):
    return f"Triangle({self.p1}, {self.p2}, {self.p3})"

  def __eq__(self, other):
      if not isinstance(other, Triangle):
          return False
      vertices_self = {self.p1, self.p2, self.p3}
      vertices_other = {other.p1, other.p2, other.p3}
      return vertices_self == vertices_other

  def __hash__(self):
      return hash(self.p1) ^ hash(self.p2) ^ hash(self.p3)


def SuperTriangle(P):
  """
  P to chmura punktów [Point(x1,y1),Point(x2,y2)...]
  """
  maxX = -float('inf')
  minX = float('inf')
  maxY = -float('inf')
  minY = float('inf')
  for p in P:
    x = p.x
    y = p.y
    if x > maxX:
      maxX = x
    if x < minX:
      minX = x
    if y > maxY:
      maxY = y
    if y < minY:
      minY = y

  dx = maxX - minX
  dy = maxY - minY
  delta = max(dx, dy)
  if delta == 0: delta = 1

  p1 = Point((minX + maxX) / 2, maxY + 20 * delta)
  p2 = Point(minX - 20 * delta, minY - delta)
  p3 = Point(maxX + 20 * delta, minY - delta)

  return Triangle(p1, p2, p3)


def is_outside(edge, p, triangle):
    val = (edge.p2.y - edge.p1.y) * (p.x - edge.p2.x) - \
          (edge.p2.x - edge.p1.x) * (p.y - edge.p2.y)
    return val > 0


def clean_super_triangle(triangulation, st):
    super_verts = {st.p1, st.p2, st.p3}
    toRemove = set()
    
    for t in triangulation:
        if t.p1 in super_verts or t.p2 in super_verts or t.p3 in super_verts:
            toRemove.add(t)

    return triangulation.difference(toRemove)


def generate_uniform(n,x,y):
    '''
    Docstring for generate_uniform
    
    :param n: points count
    :param x: maxX
    :param y: maxY
    '''
    P = [Point(np.random.uniform(0,x),np.random.uniform(0,y)) for _ in range(n)]
    return P


def json_parser(jsonFile):
    jsonFile = open(jsonFile, 'r')
    data = json.load(jsonFile)
    jsonFile.close()
    points = []
    for x,y in data:
         points.append(Point(x,y))
    return points


def draw_triangles(triangles):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for t in triangles:
        xs = [t.p1.x, t.p2.x, t.p3.x, t.p1.x]
        ys = [t.p1.y, t.p2.y, t.p3.y, t.p1.y]
        ax.plot(xs, ys, 'k-', linewidth=0.8, alpha=0.5)

    ax.set_aspect('equal')
    plt.show()

def record_state(triangulation, active_point, highlight_tris, color, title):
    snapshot = {
        'tris': list(triangulation),       
        'point': active_point,             
        'high_tris': list(highlight_tris) if highlight_tris else [],
        'color': color,
        'title': title
    }
    HISTORY.append(snapshot)


def add_point_to_triangulation(triangulation, p, start_triangle, vis=True):
    removed = set()
    visited_dfs = set()
    
    def dfs(t):
        visited_dfs.add(t)
        if t.is_in_circumcircle(p):
            removed.add(t)
            for n in t.neighbours:
                if n and n not in visited_dfs: dfs(n)
    dfs(start_triangle)


    if vis:
        record_state(triangulation, p, removed, 'red', "Usuwanie wneki (Cavity Removal)")


    boundary = {}
    for t in removed:
        for i, edge in enumerate(t.edges):
            if t.neighbours[i] not in removed: 
                boundary[edge] = t.neighbours[i]
    

    for t in removed: 
        triangulation.remove(t)

    new_triangles = []
    for edge, outer in boundary.items():
        newT = Triangle(edge.p1, edge.p2, p)
        new_triangles.append(newT)
        triangulation.add(newT)
        
        for i, e in enumerate(newT.edges):
            if e == edge:
                newT.neighbours[i] = outer
                if outer:
                    for j, oe in enumerate(outer.edges):
                        if oe == edge: 
                            outer.neighbours[j] = newT
                            break
                break
    
    shared_edges = {}
    for t in new_triangles:
        for i, e in enumerate(t.edges):
            if e in boundary: continue
            
            if e in shared_edges:
                other = shared_edges[e]
                t.neighbours[i] = other
                for j, oe in enumerate(other.edges):
                    if oe == e: 
                        other.neighbours[j] = t
                        break
            else: 
                shared_edges[e] = t

    if vis:
        record_state(triangulation, p, new_triangles, 'blue', "Wstawianie nowych (Retriangulation)")
    
    return new_triangles


def naiveSearch(points, vis=True):
    if vis:
        HISTORY.clear()
        
    triangulation = set()
    st = SuperTriangle(points)
    triangulation.add(st)

    for i, p in enumerate(points):
        found = None
        for t in triangulation:
            if t.is_inside(p): 
                found = t
                break
        
        if found:
            if vis:
                record_state(triangulation, p, [found], 'green', f"Naive Search: Punkt {i+1}")
            # Przekazujemy flagę vis dalej
            add_point_to_triangulation(triangulation, p, found, vis=vis)
        else:
            if vis:
                record_state(triangulation, p, [], 'black', f"Naive Search: ERROR Punkt {i+1}")

    final_tris = clean_super_triangle(triangulation, st)
    
    if vis:
        record_state(final_tris, None, [], 'white', "Koniec (Naive)")
        
    return final_tris


def walkingSearch(points, vis=True):
    if vis:
        HISTORY.clear()
        
    triangulation = set()
    st = SuperTriangle(points)
    triangulation.add(st)
    
    last_found = st

    for i, p in enumerate(points):
        curr = last_found
        visited = set()
        found = None
        
        # Tworzymy listę path tylko jeśli jest potrzebna do wizualizacji
        path = [] if vis else None

        while curr:
            if vis:
                path.append(curr)
                # Visualize every step of the walk
                record_state(triangulation, p, path[-1:], 'orange', f"Walking Search: Punkt {i+1} (Krok {len(path)})")

            if curr in visited:
                # Fallback on cycle
                for t in triangulation:
                    if t.is_inside(p): found = t; break
                break
            visited.add(curr)

            if curr.is_inside(p): 
                found = curr
                break
            
            moved = False
            for idx, edge in enumerate(curr.edges):
                if is_outside(edge, p, curr):
                    if curr.neighbours[idx]: 
                        curr = curr.neighbours[idx]
                        moved = True
                        break
            
            if not moved:
                 # Local minimum fallback
                 for t in triangulation:
                    if t.is_inside(p): found = t; break
                 break
        
        if found:
            if vis:
                record_state(triangulation, p, [found], 'green', f"Walking Search: Znaleziono start dla {i+1}")
            
            # Przekazujemy flagę vis dalej
            new_tris = add_point_to_triangulation(triangulation, p, found, vis=vis)
            if new_tris: 
                last_found = new_tris[0]
    
    final_tris = clean_super_triangle(triangulation, st)
    
    if vis:
        record_state(final_tris, None, [], 'white', "Koniec (Walking)")
        
    return final_tris


def render_gif(original_points, filename, duration=300):
    if not HISTORY:
        print("Brak historii do wyrenderowania.")
        return

    print(f"Renderowanie {len(HISTORY)} klatek do {filename}...")
    
    xs = [p.x for p in original_points]
    ys = [p.y for p in original_points]
    
    if not xs or not ys:
        return

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    pad_x = (max_x - min_x) * 0.15
    pad_y = (max_y - min_y) * 0.15
    if pad_x == 0: pad_x = 10
    if pad_y == 0: pad_y = 10
    
    VIEW_X = (min_x - pad_x, max_x + pad_x)
    VIEW_Y = (min_y - pad_y, max_y + pad_y)

    frames = []
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i, snap in enumerate(HISTORY):
        # Opcjonalnie: print postępu co jakiś czas
        if i % 50 == 0: print(f"Przetwarzanie klatki {i}/{len(HISTORY)}...")
        
        ax.clear()
        ax.set_title(snap['title'])
        ax.set_xlim(VIEW_X)
        ax.set_ylim(VIEW_Y)
        ax.set_aspect('equal')
        
        for t in snap['tris']:
            xs_t = [t.p1.x, t.p2.x, t.p3.x, t.p1.x]
            ys_t = [t.p1.y, t.p2.y, t.p3.y, t.p1.y]
            ax.plot(xs_t, ys_t, 'k-', linewidth=0.5, alpha=0.5)
        
        for t in snap['high_tris']:
            xs_t = [t.p1.x, t.p2.x, t.p3.x, t.p1.x]
            ys_t = [t.p1.y, t.p2.y, t.p3.y, t.p1.y]
            ax.fill(xs_t, ys_t, color=snap['color'], alpha=0.4)
            ax.plot(xs_t, ys_t, color=snap['color'], linewidth=1.5)
        
        ax.plot(xs, ys, 'b.', markersize=4, alpha=0.5)

        if snap['point']:
            ax.plot(snap['point'].x, snap['point'].y, 'ro', markersize=8)
            
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80) # Zmniejszone DPI dla szybkości
        buf.seek(0)
        frames.append(Image.open(buf))
        
    plt.close(fig)

    if frames:
        frames[0].save(filename, save_all=True, append_images=frames[1:], optimize=True, duration=duration, loop=0)
        print(f"Zapisano animację: {filename}")
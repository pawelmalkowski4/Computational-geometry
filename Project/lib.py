EPS = 0
class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def distance(self, other):
    return ((self.x - other.x)**2 + (self.y - other.y)**2)

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
#CCW > 0


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
    # Prosty test znaku pola (orientacji)
    # Zwraca True, jeśli punkt p leży po prawej stronie krawędzi (edge.p1 -> edge.p2)
    # przy założeniu, że wnętrze trójkąta jest po lewej.
    val = (edge.p2.y - edge.p1.y) * (p.x - edge.p2.x) - \
          (edge.p2.x - edge.p1.x) * (p.y - edge.p2.y)
    return val > 0 # Jeśli dodatnie, punkt jest "na zewnątrz"




# --- NOWA FUNKCJA POMOCNICZA (CORE LOGIC) ---
def add_point_to_triangulation(triangulation, p, start_triangle):
    removed = set()
    visited_dfs = set()

    # 1. DFS - szukamy trójkątów do usunięcia (nielegalnych)
    def dfs(t):
        visited_dfs.add(t)
        if t.is_in_circumcircle(p):
            removed.add(t)
            for n in t.neighbours:
                if n is not None and n not in visited_dfs:
                    dfs(n)
    
    dfs(start_triangle)

    # 2. Znajdowanie boundary (granicy wnęki)
    boundary = {} # Krawędź -> Zewnętrzny Sąsiad
    
    for t in removed:
        for i, edge in enumerate(t.edges):
            neighbor = t.neighbours[i]
            # Jeśli sąsiada nie ma w 'removed', to jest to krawędź graniczna
            if neighbor not in removed:
                boundary[edge] = neighbor

    # 3. Usuwanie starych trójkątów
    for t in removed:
        triangulation.remove(t)

    # 4. Tworzenie nowych trójkątów i łączenie z zewnętrzem
    new_triangles = []
    
    for edge, outer_neighbor in boundary.items():
        # Tworzymy nowy trójkąt łączący krawędź graniczną z nowym punktem P
        newT = Triangle(edge.p1, edge.p2, p)
        new_triangles.append(newT)
        triangulation.add(newT)

        # Łączenie z ZEWNĘTRZNYM (starym) sąsiadem
        # Musimy znaleźć, która krawędź w nowym trójkącie to 'edge'
        for i, e_new in enumerate(newT.edges):
            if e_new == edge:
                newT.neighbours[i] = outer_neighbor
                
                # Aktualizacja wskazania w starym sąsiedzie
                if outer_neighbor:
                    for j, e_old in enumerate(outer_neighbor.edges):
                        if e_old == edge:
                            outer_neighbor.neighbours[j] = newT
                            break
                break

    # 5. Łączenie NOWYCH trójkątów między sobą
    # Krawędzie wychodzące z punktu P są wspólne dla nowych trójkątów
    shared_edges = {}
    
    for t in new_triangles:
        for i, e in enumerate(t.edges):
            # Pomijamy krawędzie graniczne (już połączone z boundary)
            if e in boundary: continue

            if e in shared_edges:
                other = shared_edges[e]
                # Łączymy t i other
                t.neighbours[i] = other
                
                # Znajdź indeks w 'other'
                for j, e_other in enumerate(other.edges):
                    if e_other == e:
                        other.neighbours[j] = t
                        break
            else:
                shared_edges[e] = t
    
    return new_triangles # Zwracamy listę nowych trójkątów


def clean_super_triangle(triangulation, st):
    # Tworzymy zbiór wierzchołków super-trójkąta dla szybkiego sprawdzania
    super_verts = {st.p1, st.p2, st.p3}
    toRemove = set()
    
    for t in triangulation:
        if t.p1 in super_verts or t.p2 in super_verts or t.p3 in super_verts:
            toRemove.add(t)
            
    # Czyścimy referencje sąsiadów w pozostawionych trójkątach 
    # (żeby nie wskazywały na usunięte), choć w Pythoie GC to załatwi, 
    # dla czystości topologicznej można by ustawić na None.
    
    return triangulation.difference(toRemove)


# --- METODA 1: NAIVE SEARCH ---
def naiveSearch(points):
    triangulation = set()
    st = SuperTriangle(points)
    triangulation.add(st)

    for p in points:
        start_triangle = None
        # Proste przeszukiwanie liniowe
        for t in triangulation:
            if t.is_inside(p):
                start_triangle = t
                break
        
        # Jeśli z jakiegoś powodu (błędy numeryczne) nie znaleziono, 
        # można spróbować przeszukać ponownie z większym marginesem, 
        # ale w idealnym przypadku zawsze się znajdzie (dzięki SuperTriangle)
        
        if start_triangle:
            add_point_to_triangulation(triangulation, p, start_triangle)

    return clean_super_triangle(triangulation, st)


# --- METODA 2: WALKING SEARCH ---
def walkingSearch(points):
    triangulation = set()
    st = SuperTriangle(points)
    triangulation.add(st)
    
    last_found = st

    for p in points:
        curr = last_found
        visited_walk = set()
        found_triangle = None
        
        # Algorytm spaceru
        while curr is not None:
            if curr in visited_walk:
                # Zapętlenie (rzadkie, ale możliwe przy błędach numerycznych).
                # Fallback do metody naiwnej.
                for t in triangulation:
                    if t.is_inside(p):
                        found_triangle = t
                        break
                break
            
            visited_walk.add(curr)
            
            if curr.is_inside(p):
                found_triangle = curr
                break
            
            # Decyzja nawigacyjna: przez którą krawędź przejść?
            moved = False
            for i, edge in enumerate(curr.edges):
                if is_outside(edge, p, curr):
                    next_tri = curr.neighbours[i]
                    if next_tri is not None:
                        curr = next_tri
                        moved = True
                        break
            
            if not moved:
                # Punkt jest na zewnątrz obecnego trójkąta, ale nie ma sąsiada w tym kierunku.
                # Może się zdarzyć na granicy SuperTriangle lub przy błędach numerycznych.
                # Fallback.
                for t in triangulation:
                    if t.is_inside(p):
                        found_triangle = t
                        break
                break

        if found_triangle:
            new_tris = add_point_to_triangulation(triangulation, p, found_triangle)
            if new_tris:
                # Optymalizacja: następny punkt prawdopodobnie będzie blisko tego
                last_found = new_tris[0]
    print(1)
    return clean_super_triangle(triangulation, st)

import numpy as np
import time
import matplotlib.pyplot as plt
ns = []
ws = []
ys = []
for i,n in enumerate([100,200,300,400,800,1600,3200,6600,7800,10000,20000]):
    Points = [Point(np.random.uniform(-100*(i+1),100*(i+1)),np.random.uniform(-100*(i+1),100*(i+1))) for _ in range(n)]
    start = time.perf_counter()
    r1=naiveSearch(Points)
    end = time.perf_counter()
    ns.append(end-start)

    start = time.perf_counter()
    r2 = walkingSearch(Points)
    end = time.perf_counter()
    ws.append(end-start)
    ys.append(n)

    print(r1 == r2)
    

plt.plot(ys,ns, c = 'red')
plt.plot(ys,ws, c='blue')
plt.xlabel('n')
plt.ylabel('t[ms]')
plt.show()
import math
import re
from collections import defaultdict
from itertools import combinations
from operator import attrgetter

from dataclasses import dataclass


@dataclass(hash=True)
class Vector:
    x: int
    y: int
    z: int

    @property
    def distance(self):
        return abs(self.x) + abs(self.y) + abs(self.z)
    

def solve_quadratic(p1, p2, v1, v2, a1, a2):
    a = (a1 - a2) / 2
    b = (v1 - v2) + a
    c = p1 - p2
    discriminant = (b ** 2) - (4 * a * c)
    if discriminant < 0:
        # no solutions
        return ()
    if discriminant == 0:
        return (-b / (2 * a)),
    return (
        (-b + math.sqrt(discriminant)) / (2 * a),
        (-b - math.sqrt(discriminant)) / (2 * a)
    )

def solve_linear(p1, p2, v1, v2):
    return (p2 - p1) / (v1 - v2)


def solve(p1, p2, v1, v2, a1, a2):
    """Find the positive integer points in time where P1 and P2 intersect
    
    Returns a set with points in time.
    
    """
    if a1 == a2:
        ts = solve_linear(p1, p2, v1, v2),
    else:
        ts = solve_quadratic(p1, p2, v1, v2, a1, a2)
    return {int(round(t)) for t in ts if t > 0 and math.isclose(t, round(t))}


@dataclass(hash=True)
class Particle:
    id: int
    p: Vector
    v: Vector
    a: Vector

    @classmethod
    def from_line(cls, idx, line, _d=re.compile(r'-?\d+')):
        px, py, pz, vx, vy, vz, ax, ay, az = map(int, _d.findall(line))
        return cls(idx, Vector(px, py, pz), Vector(vx, vy, vz), Vector(ax, ay, az))
    
    def __and__(self, other):
        # find if two particles will collide in a future point in time
        solutions = None
        for c in 'xyz':
            p1, p2, v1, v2, a1, a2 = (
                getattr(getattr(ob, v), c) for v in 'pva'
                for ob in (self, other))
            if a1 == a2 and v1 == v2:
                # parallel paths, always matching if starting position is equal
                if p1 == p2:
                    continue
                # positions not equal, will never cross
                return False, None
            if solutions is None:
                solutions = solve(p1, p2, v1, v2, a1, a2)
            else:
                solutions &= solve(p1, p2, v1, v2, a1, a2)
            if not solutions:
                return False, None
        return True, min(solutions)


def read_particles(lines):
    return [Particle.from_line(i, l) for i, l in enumerate(lines)]


def find_closest(particles):
    # Assumption: the lowest absolute acceleration and velocity will win
    return min(particles, key=lambda p: (p.a.distance, p.v.distance, p.p.distance))


def eliminate_collisions(particles):
    collisions = defaultdict(set)
    for p1, p2 in combinations(particles, 2):
        collide, time = p1 & p2
        if not collide:
            continue
        collisions[time] |= {p1, p2}

    eliminated = None
    for time, collided in sorted(collisions.items()):
        if not eliminated:
            eliminated = set(collided)
        else:
            for p1, p2 in combinations(collided - eliminated, 2):
                if (p1 & p2)[0]:
                    eliminated |= {p1, p2}
    return len(particles) - len(eliminated)

test_particles = read_particles('''p=< 3,0,0>, v=< 2,0,0>, a=<-1,0,0>
p=< 4,0,0>, v=< 0,0,0>, a=<-2,0,0>
'''.splitlines())
assert find_closest(test_particles).id == 0

test_particles = read_particles('''p=<-6,0,0>, v=< 3,0,0>, a=< 0,0,0>    
p=<-4,0,0>, v=< 2,0,0>, a=< 0,0,0>
p=<-2,0,0>, v=< 1,0,0>, a=< 0,0,0>
p=< 3,0,0>, v=<-1,0,0>, a=< 0,0,0>
'''.splitlines())
assert eliminate_collisions(test_particles) == 1

with open('inputs/day20.txt') as day20:
    particles = read_particles(day20)

print('Part 1:', find_closest(particles).id)

print('Part 2:', eliminate_collisions(particles))


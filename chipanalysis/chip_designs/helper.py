
try:
    # Standalone Python (pip install klayout)
    import klayout.db as db
    import klayout.lay as lay  # GUI side (Qt) – only needed if you want a window
    pya = db                   # optional: keep "pya" name compatible with macros
except Exception:
    # Inside KLayout macro runner
    import pya                 # provided by KLayout
    db = pya
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass, replace, asdict

CONV = 1000 # GLOBAL VARIABLE: UNIT CONVERSTION FROM um TO nm


def linspace(a, b, n):
    if n < 2:
        return b
    diff = (float(b) - a) / (n - 1)
    return [diff * i + a for i in range(n)]


def linspace2(a, b, diff):
    diffspace = [a]
    i = 1
    while diffspace[len(diffspace) - 1] < b:
        diffspace.append(a + diff * i)
        i = i + 1
    return diffspace


# CORRECT COORDINATES BASED ON FIELD, LIST CONTAINS central x y coordinates, and x_length and y_length.
def CX(corx,x=0):
    xcor = (corx + x) * CONV
    return xcor


def CY(cory,y=0):
    ycor = (cory + y) * CONV
    return ycor


# CICRLE SHAPE: INLET/OUTLET/WAFER/PILLARS
def Circle(xcor, ycor, diameter, npoints,x=0,y=0):
    xcor = CX(xcor,x)
    ycor = CY(ycor,y)
    radius = diameter * CONV / 2
    npoint = 100
    angles = linspace(0, 2 * math.pi, npoints + 1)[0:-1]
    points = []
    for ind, angle in enumerate(angles):
        points.append(pya.Point(radius * math.cos(angle) + xcor, radius * math.sin(angle) + ycor))
    CIRCLE = pya.SimplePolygon(points)
    return CIRCLE


# CICRLE SHAPE: INLET/OUTLET/WAFER/PILLARS
def Circle_coor(xcor, ycor, diameter, npoints):
    xcor = CX(xcor)
    ycor = CY(ycor)
    radius = diameter * CONV / 2
    npoint = 100
    angles = linspace(0, 2 * math.pi, npoints + 1)[0:-1]
    points = []
    for ind, angle in enumerate(angles):
        points.append([radius * math.cos(angle) + xcor, radius * math.sin(angle) + ycor])
    return points


# WORK AREA OF CHIP
def Region(xcor, ycor, xlength, ylength,x=0,y=0):
    xcor = CX(xcor,y)
    ycor = CY(ycor,y)
    xlength = xlength * CONV
    ylength = ylength * CONV
    Chip = pya.Box(xcor - 0.5 * xlength, ycor - 0.5 * ylength, xcor + 0.5 * xlength, ycor + 0.5 * ylength)
    return Chip


# WORK AREA OF CHIP
def Region_coor(xcor, ycor, xlength, ylength):
    xcor = CX(xcor)
    ycor = CY(ycor)
    xlength = xlength * CONV
    ylength = ylength * CONV
    points = []
    points.append([xcor - 0.5 * xlength, ycor - 0.5 * ylength])
    points.append([xcor - 0.5 * xlength, ycor + 0.5 * ylength])
    points.append([xcor + 0.5 * xlength, ycor + 0.5 * ylength])
    points.append([xcor + 0.5 * xlength, ycor - 0.5 * ylength])
    return points


def Rotate(x_ref, y_ref, angle, coordinates):
    new_coordinates = list()
    for i in range(len(coordinates)):
        xcor = coordinates[i][0]
        ycor = coordinates[i][1]
        angle_radians = math.radians(angle)
        x_prime = xcor - CX(x_ref)
        y_prime = ycor - CY(y_ref)
        x_rotated_prime = x_prime * math.cos(angle_radians) - y_prime * math.sin(angle_radians)
        y_rotated_prime = x_prime * math.sin(angle_radians) + y_prime * math.cos(angle_radians)
        x_new = x_rotated_prime + CX(x_ref)
        y_new = y_rotated_prime + CY(y_ref)
        new_coordinates.append(pya.Point(x_new, y_new))
        polygon = pya.SimplePolygon(new_coordinates)
    return polygon


def Trapezoid(xcor, ycor, a_length, b_length, h_height):
    xcor = CX(xcor)
    ycor = CY(ycor)
    alength = a_length * CONV
    blength = b_length * CONV
    hheigth = h_height * CONV

    points = []
    points.append([xcor - 0.5 * alength, ycor + 0.5 * hheigth])  # a1
    points.append([xcor + 0.5 * alength, ycor + 0.5 * hheigth])  # a2
    points.append([xcor + 0.5 * blength, ycor - 0.5 * hheigth])  # b2
    points.append([xcor - 0.5 * blength, ycor - 0.5 * hheigth])  # b1

    TRAPEZOID = pya.SimplePolygon(points)
    return TRAPEZOID


def Trapezoid_coor(xcor, ycor, a_length, b_length, h_height, pillar, pillar_size, pillar_distance):
    xcor = CX(xcor)
    ycor = CY(ycor)
    alength = a_length * CONV
    blength = b_length * CONV
    hheigth = h_height * CONV

    points = []
    points.append([xcor - 0.5 * alength, ycor + 0.5 * hheigth])  # a1
    points.append([xcor + 0.5 * alength, ycor + 0.5 * hheigth])  # a2
    points.append([xcor + 0.5 * blength, ycor - 0.5 * hheigth])  # b2
    points.append([xcor - 0.5 * blength, ycor - 0.5 * hheigth])  # b1

    return points


def Polygon_house(xcor, ycor, funnel_length, funnel_height, house_height, house_length, pillar, pillar_size,
                  pillar_distance, angle):
    region_coor = Polygon_house_coor(xcor, ycor, funnel_length, funnel_height, house_height, house_length)
    region = Rotate(xcor, ycor, angle, region_coor)
    GrowthChamber = pya.Region(region)

    xlength = house_length + funnel_length
    ylength = funnel_height + house_height + 1000

    if pillar == 1:
        # make array of circles
        a = pillar_distance * CONV  # DISTANCE BETWEEN PILLARS
        n = round(1.5 * max([xlength, ylength]) / pillar_distance)  # NUMBER OF PILLARS
        print("number of pillars in house: ", n)
        dx = 2 * a * math.sin(math.radians(30))  # horizontal distance between hexagons
        dy = a * math.cos(math.radians(30))  # vertical distance between hexagons
        coordinates = list()
        for i_x in range(n):
            for i_y in range(n):
                yhex = i_y * dy
                if i_y % 2 == 0:
                    xhex = i_x * dx + 0.5 * dx
                if i_y % 2 == 1:
                    xhex = i_x * dx
                if xhex < ((xlength - pillar_size) * CONV) and yhex < ((ylength - pillar_size) * CONV):
                    coordinates.append([xhex, yhex])

        x_cors = [item[0] for item in coordinates]
        y_cors = [item[1] for item in coordinates]

        circles = list()
        for i in range(len(coordinates)):
            xcircle = (coordinates[i][0] - min(x_cors)) / CONV - 0.5 * max(x_cors) / CONV + (xcor)
            ycircle = (coordinates[i][1] - min(y_cors)) / CONV - 0.5 * max(y_cors) / CONV + (
                        ycor - 20)  # -20 to move the pillars inside the polygon house to not have cut pillars at the edges
            sub_coordinates = Circle_coor(xcircle, ycircle, pillar_size, 100)
            circles.append(Rotate(xcor, ycor, angle, sub_coordinates))

        print(circles)

        for i in range(len(circles)):
            GrowthChamber = GrowthChamber - pya.Region(circles[i])

    return GrowthChamber


def Polygon_house_coor(xcor, ycor, funnel_length, funnel_height, house_height, house_length):
    xcor = CX(xcor)
    ycor = CY(ycor)
    funnellength = funnel_length * CONV
    funnelheight = funnel_height * CONV
    househeigth = house_height * CONV
    houselength = house_length * CONV

    points = []
    points.append([xcor - 0.5 * funnellength, ycor + 0.5 * househeigth + funnelheight])  # a
    points.append([xcor + 0.5 * funnellength, ycor + 0.5 * househeigth + funnelheight])  # b
    points.append([xcor + 0.5 * houselength, ycor + 0.5 * househeigth])  # c
    points.append([xcor + 0.5 * houselength, ycor - 0.5 * househeigth])  # d
    points.append([xcor - 0.5 * houselength, ycor - 0.5 * househeigth])  # e
    points.append([xcor - 0.5 * houselength, ycor + 0.5 * househeigth])  # f

    return points


def Barrier(xcor, ycor, xlength, type_wall, gap, size, layers, angle,margin):
    ylength = layers * (size + gap) - gap + margin
    if type_wall == "block":
        d = size + gap
        if layers == 1:
            nx = math.ceil(1.1 * xlength / d)  # NUMBER OF BLOCKS
        if layers > 1:
            nx = math.ceil(2 * xlength / d)  # NUMBER OF BLOCKS
        d = d * CONV

        coordinates = list()
        for i_x in range(nx):
            for i_y in range(layers):
                yreg = i_y * d
                if i_y % 2 == 0:
                    xreg = i_x * d + 0.5 * d
                if i_y % 2 == 1:
                    xreg = i_x * d
                coordinates.append([xreg, yreg])

    if type_wall == "pillar":
        a = (gap + size) * CONV  # DISTANCE BETWEEN PILLARS
        if layers == 1:
            nx = math.ceil(1.1 * xlength / (gap + size))  # NUMBER OF PILLARS
        if layers > 1:
            nx = math.ceil(2 * xlength / (gap + size))  # NUMBER OF PILLARS
        dx = 2 * a * math.sin(math.radians(30))  # horizontal distance between hexagons
        dy = a * math.cos(math.radians(30))  # vertical distance between hexagons
        # ylength = 2*(gap+size)*math.cos(math.radians(30))+size+MARGIN

        coordinates = list()
        for i_x in range(nx):
            for i_y in range(layers):
                yhex = i_y * dy
                if i_y % 2 == 0:
                    xhex = i_x * dx + 0.5 * dx
                if i_y % 2 == 1:
                    xhex = i_x * dx
                coordinates.append([xhex, yhex])

    region_coor = Region_coor(xcor, ycor, xlength, ylength)
    region = Rotate(xcor, ycor, angle, region_coor)
    GrowthChamber = pya.Region(region)

    x_cors = [item[0] for item in coordinates]
    y_cors = [item[1] for item in coordinates]

    objects = list()
    for i in range(len(coordinates)):
        xobject = (coordinates[i][0] - sum(x_cors) / len(x_cors)) / CONV + xcor
        yobject = (coordinates[i][1] - sum(y_cors) / len(y_cors)) / CONV + ycor

        if type_wall == "pillar":
            sub_coordinates = Circle_coor(xobject, yobject, size, 100)
        if type_wall == "block":
            sub_coordinates = Region_coor(xobject, yobject, size, size)
        objects.append(Rotate(xcor, ycor, angle, sub_coordinates))

    for i in range(len(objects)):
        GrowthChamber = GrowthChamber - pya.Region(objects[i])

    return GrowthChamber


def Barrier_Long(xcor, ycor, xlength, block_width, block_height, gap, angle,margin):
    ylength = block_height + margin
    d = block_width + gap
    nx = math.ceil(1.1 * xlength / d)  # NUMBER OF BLOCKS
    d = d * CONV

    coordinates = list()
    for i_x in range(nx):
        xreg = i_x * d
        coordinates.append([xreg, 0])

    region_coor = Region_coor(xcor, ycor, xlength, ylength)
    region = Rotate(xcor, ycor, angle, region_coor)
    GrowthChamber = pya.Region(region)

    x_cors = [item[0] for item in coordinates]
    y_cors = [item[1] for item in coordinates]

    objects = list()
    for i in range(len(coordinates)):
        xobject = (coordinates[i][0] - sum(x_cors) / len(x_cors)) / CONV + xcor
        yobject = coordinates[i][1] / CONV + ycor
        sub_coordinates = Region_coor(xobject, yobject, block_width, block_height)
        objects.append(Rotate(xcor, ycor, angle, sub_coordinates))

    for i in range(len(objects)):
        GrowthChamber = GrowthChamber - pya.Region(objects[i])

    return GrowthChamber


def Chamber(xcor, ycor, xlength, ylength, pillar, pillar_size, pillar_distance, angle):
    region_coor = Region_coor(xcor, ycor, xlength, ylength)
    region = Rotate(xcor, ycor, angle, region_coor)
    GrowthChamber = pya.Region(region)

    if pillar == 1:
        # make array of circles
        a = pillar_distance * CONV  # DISTANCE BETWEEN PILLARS
        n = round(1.5 * max([xlength, ylength]) / pillar_distance)  # NUMBER OF PILLARS
        print("number of pillars in chamber: ", n)
        dx = 2 * a * math.sin(math.radians(30))  # horizontal distance between hexagons
        dy = a * math.cos(math.radians(30))  # vertical distance between hexagons
        coordinates = list()
        for i_x in range(n):
            for i_y in range(n):
                yhex = i_y * dy
                if i_y % 2 == 0:
                    xhex = i_x * dx + 0.5 * dx
                if i_y % 2 == 1:
                    xhex = i_x * dx
                if xhex < ((xlength - pillar_size) * CONV) and yhex < ((ylength - pillar_size) * CONV):
                    coordinates.append([xhex, yhex])

        x_cors = [item[0] for item in coordinates]
        y_cors = [item[1] for item in coordinates]

        circles = list()
        for i in range(len(coordinates)):
            xcircle = (coordinates[i][0] - min(x_cors)) / CONV - 0.5 * max(x_cors) / CONV + xcor
            ycircle = (coordinates[i][1] - min(y_cors)) / CONV - 0.5 * max(y_cors) / CONV + ycor
            sub_coordinates = Circle_coor(xcircle, ycircle, pillar_size, 100)
            circles.append(Rotate(xcor, ycor, angle, sub_coordinates))

        for i in range(len(circles)):
            GrowthChamber = GrowthChamber - pya.Region(circles[i])

    return GrowthChamber


def Chamber_with_gaps(xcor, ycor, xlength, ylength, pillar, pillar_size, pillar_distance, angle, gap, gap_num, gap_x,
                      gap_y, gap_xlength, gap_ylength):
    region_coor = Region_coor(xcor, ycor, xlength, ylength)
    region = Rotate(xcor, ycor, angle, region_coor)
    GrowthChamber = pya.Region(region)

    if gap == 1:
        for k in range(gap_num):
            gap_coor = Region_coor(gap_x[k], gap_y[k], gap_xlength, gap_ylength)
            gap = (Rotate(xcor, ycor, angle, gap_coor))
            GrowthChamber = GrowthChamber - pya.Region(gap)

    if pillar == 1:
        # make array of circles
        a = pillar_distance * CONV  # DISTANCE BETWEEN PILLARS
        n = round(1.5 * max([xlength, ylength]) / pillar_distance)  # NUMBER OF PILLARS
        print("number of pillars in chamber: ", n)
        dx = 2 * a * math.sin(math.radians(30))  # horizontal distance between hexagons
        dy = a * math.cos(math.radians(30))  # vertical distance between hexagons
        coordinates = list()
        for i_x in range(n):
            for i_y in range(n):
                yhex = i_y * dy
                if i_y % 2 == 0:
                    xhex = i_x * dx + 0.5 * dx
                if i_y % 2 == 1:
                    xhex = i_x * dx
                if xhex < ((xlength - pillar_size) * CONV) and yhex < ((ylength - pillar_size) * CONV):
                    coordinates.append([xhex, yhex])

        x_cors = [item[0] for item in coordinates]
        y_cors = [item[1] for item in coordinates]

        circles = list()
        for i in range(len(coordinates)):
            xcircle = (coordinates[i][0] - min(x_cors)) / CONV - 0.5 * max(x_cors) / CONV + xcor
            ycircle = (coordinates[i][1] - min(y_cors)) / CONV - 0.5 * max(y_cors) / CONV + ycor
            sub_coordinates = Circle_coor(xcircle, ycircle, pillar_size, 100)
            circles.append(Rotate(xcor, ycor, angle, sub_coordinates))

        for i in range(len(circles)):
            GrowthChamber = GrowthChamber - pya.Region(circles[i])

    return GrowthChamber


def Channel(xcor1, ycor1, xcor2, ycor2, width):
    xcor1 = CX(xcor1)
    ycor1 = CY(ycor1)
    xcor2 = CX(xcor2)
    ycor2 = CY(ycor2)
    width = width * CONV

    points = [pya.Point(xcor1, ycor1), pya.Point(xcor2, ycor2)]
    line = pya.Path(points, width, 0.5 * width, 0.5 * width, True)
    line = line.round_corners(50 * width, 50)
    line = line.polygon()
    return line


# MIXING CHANNEL, WHEN MERGING TO INLET CHANNELS
def Mix(xcor1, ycor1, xcor2, ycor2, number, width):
    nothing = 0
    return nothing


# ADD TEXT TO CHIP TO COORDINATE YOURSELF
def Name(xcor, ycor, text, angle,
         size=50):  # trabs rotates the test "RO" with 0 degrees, "R180" with 180 degrees, etc. only 90 degrees steps

    REGION = pya.TextGenerator.default_generator().text(text, 0.001, 0.001 * CONV * size)
    REGION = REGION.moved(CX(xcor), CY(ycor))
    REGION = REGION + Region(xcor, ycor - 10, 5, 5)

    return REGION


# ADD COORDINATES BELOW CHAMBER
def Coordinates(xcor, ycor, width, mode, size=25):
    if mode == 0:
        delta = 10
    if mode == 1:
        delta = -10 - size
    xname = linspace2(0, width, 100)
    xcors = [xcor + i for i in xname]
    region = pya.TextGenerator.default_generator().text(str(xname[1]), 0.001, 0.001 * CONV * size)
    region = region.moved(CX(xcors[1]), CY(ycor + delta))
    REGION = region + Region(xcors[1], ycor, 5, 5)
    for i in range(len(xcors)):
        if i == 0:
            continue

        else:
            region = pya.TextGenerator.default_generator().text(str(xname[i]), 0.001, 0.001 * CONV * size)
            region = region.moved(CX(xcors[i]), CY(ycor + delta))
            region = region + Region(xcors[i], ycor, 5, 5)
            REGION = REGION + region

    return REGION


# REFERENCE CROSSES FOR ALIGNMENT OF CHIP
def Reference(xcor, ycor):
    nothing = 0
    return nothing


# REMOVE ANY OVERLAPPING OBJECTS/SHAPES IN LAYER OF INTEREST
def MergeLayers(top,layout,layers):
    for layer in layers:
        shapes = pya.Region(top.begin_shapes_rec(layer))
        shapes.merge()
        layout.clear_layer(layer)
        top.shapes(layer).insert(shapes)



# ALIGNMENT MARKERS FOR TWO LAYERS
def Pattern(xcor, ycor, form,x=0,y=0):
    if form == 0:
        region1 = Region(xcor + 60, ycor + 100, 40, 120,x,y)
        region2 = Region(xcor + 60, ycor + 100, 120, 40,x,y)
        REGION = pya.Region(region1) + pya.Region(region2)
        bx = [160, 160, 310]
        by = [150, 0, 0]
        width = [10, 20, 40]
        for i in range(len(width)):
            tx = [bx[i] + 0.5 * width[i], bx[i] + 2.5 * width[i], bx[i] + 4.5 * width[i]]
            ty = [by[i] + 0.5 * width[i], by[i] + 2.5 * width[i], by[i] + 4.5 * width[i]]
            for ix in range(len(tx)):
                for iy in range(len(ty)):
                    region3 = Region(xcor + tx[ix], ycor + ty[iy], width[i], width[i],x,y)
                    REGION = REGION + pya.Region(region3)
        return REGION

    if form == 1:
        bx = [-40, 160, 160, 310]
        by = [0, 150, 0, 0]
        width = [40, 10, 20, 40]
        region1 = Region(xcor + 20, ycor + 60, 40, 40)
        REGION = pya.Region(region1)
        for i in range(len(width)):
            tx = [bx[i] + 1.5 * width[i], bx[i] + 3.5 * width[i]]
            ty = [by[i] + 1.5 * width[i], by[i] + 3.5 * width[i]]
            for ix in range(len(tx)):
                for iy in range(len(ty)):
                    REGION = REGION + pya.Region(Region(xcor + tx[ix], ycor + ty[iy], width[i], width[i]))

        return REGION


def Markers(top,layer1,layer2,layer3,x,y):
    top.shapes(layer1).insert(Pattern(1190 + 300, 300, 0))
    top.shapes(layer3).insert(Pattern(1190 - 300, -300, 0))
    top.shapes(layer3).insert(Pattern(-1250 + 300, 300, 0))
    top.shapes(layer2).insert(Pattern(1190 - 300, -300, 0))
    top.shapes(layer2).insert(Pattern(-1250 + 300, 300, 0))
    top.shapes(layer1).insert(Pattern(-1250, 0, 0))

    region1 = Region(190, 100, 9000, 300,x,y)
    region2 = Region(190, 100, 300, 3000,x,y)
    region3 = Pattern(-1250, 0, 1)
    region4 = Pattern(1190, 0, 1)
    REGION = pya.Region(region1) + pya.Region(region2) - region3 - region4
    top.shapes(layer3).insert(REGION)

    region1 = Region(190, 100, 9000, 300)
    region2 = Region(190, 100, 300, 3000)
    region3 = Pattern(-1250, 0, 1)
    region4 = Pattern(1190, 0, 1)
    REGION = pya.Region(region1) + pya.Region(region2) - region3 - region4
    top.shapes(layer2).insert(REGION)

    region1 = Region(-110, -200, 9000, 300)
    region2 = Region(-110, -200, 300, 3000)
    region3 = Pattern(-1250 - 300, -300, 1)
    region4 = Pattern(1190 - 300, -300, 1)
    REGION = pya.Region(region1) + pya.Region(region2) - region3 - region4
    top.shapes(layer1).insert(REGION)

def fill_with_ppa_grid(layout,
                       cell,
                       target_geom,
                       layer_index,
                       channel_width_um=4.0,
                       spacing_um=50.0,
                       two_directions=True,
                       rotate_deg=0):
    """
    Fill 'target_geom' with a Porous Plate Analogue (PPA) grid: a set of
    parallel rectangular channels (4 µm wide by default) separated by the
    specified spacing. Optionally draws a second, orthogonal set to make a grid.

    Since you're making a negative mold, this function *draws the channels*
    as positive polygons on the specified layer (e.g., layer 1 = 'ground').

    Parameters
    ----------
    layout : pya.Layout
    cell   : pya.Cell               # where to draw
    target_geom : one of            # the region to fill (any of):
        - pya.Region
        - pya.Shape
        - pya.Polygon
        - pya.Box
        - int layer index (e.g., 'ground' from layout.layer(...))
          -> uses all shapes from that layer within 'cell'
    layer_index : int               # e.g., 'ground' (layout.layer(1,0))
    channel_width_um : float        # channel width, µm (default 4.0)
    spacing_um       : float        # *gap* between channels, µm (default 50.0)
    two_directions   : bool         # draw both vertical and horizontal sets
    rotate_deg       : float        # rotate the whole grid (about bbox center)

    Returns
    -------
    pya.Region   # the polygons that were inserted
    """
    pya = db  # keep naming consistent with your script

    # --- utilities
    dbu = layout.dbu  # µm per database unit (usually 0.001)
    def um2dbu(x): return int(round(x / dbu))

    def to_region(obj):
        """Convert various inputs to a Region."""
        if isinstance(obj, pya.Region):
            return obj.dup()
        reg = pya.Region()
        if isinstance(obj, pya.Shape):
            if obj.is_box():
                reg.insert(obj.box())
            elif obj.is_polygon():
                reg.insert(obj.polygon)
            elif obj.is_path():
                reg.insert(obj.path().polygon())
            else:
                # Try generic conversion (may raise if unsupported)
                reg.insert(obj.polygon)
            return reg
        if isinstance(obj, pya.Polygon):
            reg.insert(obj)
            return reg
        if isinstance(obj, pya.Box):
            reg.insert(obj)
            return reg
        # If a layer index (int) or pya.LayerIndex was provided, pull all shapes from it
        if isinstance(obj, int):
            reg = pya.Region(cell.begin_shapes_rec(obj))
            return reg
        raise TypeError("Unsupported target_geom type for PPA fill: {}".format(type(obj)))

    target_reg = to_region(target_geom)
    if target_reg.is_empty():
        return pya.Region()

    # --- dimensions in dbu
    w_dbu = um2dbu(channel_width_um)
    gap_dbu = um2dbu(spacing_um)
    pitch = w_dbu + gap_dbu

    # --- build stripes covering the target bbox, then clip
    bbox = target_reg.bbox()
    # Slight padding so we don't miss edges due to rounding
    pad = pitch + w_dbu
    ext = pya.Box(bbox.left - pad, bbox.bottom - pad, bbox.right + pad, bbox.top + pad)

    stripes_vert = pya.Region()
    x = ext.left
    # Align first stripe so grid is stable w.r.t. absolute origin
    # (so identical params repeat consistently across different shapes)
    # Shift x to the nearest multiple of pitch relative to 0
    if pitch > 0:
        x = ext.left - ((ext.left) % pitch)

    while x <= ext.right:
        stripe = pya.Box(x, ext.bottom, x + w_dbu, ext.top)
        stripes_vert.insert(stripe)
        x += pitch

    stripes_horiz = pya.Region()
    if two_directions:
        y = ext.bottom
        if pitch > 0:
            y = ext.bottom - ((ext.bottom) % pitch)
        while y <= ext.top:
            stripe = pya.Box(ext.left, y, ext.right, y + w_dbu)
            stripes_horiz.insert(stripe)
            y += pitch

    grid_reg = stripes_vert
    if two_directions:
        grid_reg = grid_reg + stripes_horiz  # union of both directions

    # --- optional rotation of the grid (about the target bbox center)
    if rotate_deg % 360 != 0:
        cx = (bbox.left + bbox.right) / 2.0
        cy = (bbox.bottom + bbox.top) / 2.0
        cx_d, cy_d = cx * dbu, cy * dbu

        rot_rad = math.radians(rotate_deg)
        cos_r, sin_r = math.cos(rot_rad), math.sin(rot_rad)

        def rot_point(x, y):
            dx, dy = x - cx_d, y - cy_d
            return pya.DPoint(cx_d + dx * cos_r - dy * sin_r,
                              cy_d + dx * sin_r + dy * cos_r)

        new_reg = pya.Region()
        for poly_wrapped in grid_reg.each_merged():
            # unwrap PolygonWithProperties if needed
            poly = getattr(poly_wrapped, "polygon", poly_wrapped)

            # get vertices in integer dbu, convert to D, rotate, back to I
            pts_i = [pt for pt in poly.each_point_hull()]
            pts_d = [pya.DPoint(pt.x * dbu, pt.y * dbu) for pt in pts_i]
            rot_d = [rot_point(pt.x, pt.y) for pt in pts_d]
            rot_i = [pya.Point(int(round(pt.x / dbu)), int(round(pt.y / dbu))) for pt in rot_d]

            new_reg.insert(pya.Polygon(rot_i))

        grid_reg = new_reg

    # --- clip stripes to target region
    grid_in_target = grid_reg & target_reg

    # --- write to layer
    cell.shapes(layer_index).insert(grid_in_target)

    return grid_in_target

import math

def fill_with_variable_channels(layout,
                                cell,
                                target_geom,
                                layer_index,
                                min_width_um=4.0,
                                max_width_um=20.0,
                                gap_um=4.0,
                                orientation="vertical"):
    """
    Fill 'target_geom' with a set of parallel channels whose widths vary
    linearly from min_width_um to max_width_um. Channels are oriented
    either vertically (default) or horizontally.

    Spacing rule (constant gap):
        gap(i, i+1) = gap_um

    The number of channels is chosen as the maximum that fits into the
    target's bounding box along the chosen orientation.

    Returns
    -------
    pya.Region   # the polygons that were inserted
    """

    pya = db

    if min_width_um <= 0 or max_width_um <= 0:
        raise ValueError("Channel widths must be > 0 µm")
    if gap_um < 0:
        raise ValueError("gap_um must be >= 0 µm")

    if min_width_um > max_width_um:
        min_width_um, max_width_um = max_width_um, min_width_um

    dbu = layout.dbu  # µm per database unit

    def um2dbu(x: float) -> int:
        return int(round(x / dbu))

    def to_region(obj):
        if isinstance(obj, pya.Region):
            return obj.dup()
        reg = pya.Region()
        if isinstance(obj, pya.Shape):
            if obj.is_box():
                reg.insert(obj.box())
            elif obj.is_polygon():
                reg.insert(obj.polygon)
            elif obj.is_path():
                reg.insert(obj.path().polygon())
            else:
                reg.insert(obj.polygon)
            return reg
        if isinstance(obj, pya.Polygon):
            reg.insert(obj)
            return reg
        if isinstance(obj, pya.Box):
            reg.insert(obj)
            return reg
        if isinstance(obj, int):
            return pya.Region(cell.begin_shapes_rec(obj))
        raise TypeError("Unsupported target_geom type: {}".format(type(obj)))

    target_reg = to_region(target_geom)
    if target_reg.is_empty():
        return pya.Region()

    bbox = target_reg.bbox()
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    # Length available along the axis along which channels are placed side-by-side
    if orientation == "vertical":
        axis_min_dbu = bbox.left
        axis_max_dbu = bbox.right
    else:
        axis_min_dbu = bbox.bottom
        axis_max_dbu = bbox.top

    length_dbu = axis_max_dbu - axis_min_dbu
    length_um = length_dbu * dbu

    a = float(min_width_um)
    b = float(max_width_um)
    g = float(gap_um)

    if length_um <= 0 or length_um < a:
        return pya.Region()

    # --- total span for N channels with constant gap and linear widths a->b
    # widths sum = N*(a+b)/2
    # total span = sum(widths) + (N-1)*g
    def total_extent_um(N: int) -> float:
        if N <= 0:
            return 0.0
        if N == 1:
            return a
        return (N * (a + b) / 2.0) + (N - 1) * g

    # Find maximum N that fits (simple growth; N is usually not huge)
    N = 1
    while total_extent_um(N + 1) <= length_um + 1e-9:
        N += 1

    if N <= 0:
        return pya.Region()

    # --- build width list (in µm), linearly spaced
    if N == 1:
        widths_um = [a]
    else:
        widths_um = []
        for i in range(N):
            t = float(i) / float(N - 1)  # 0..1
            widths_um.append(a + t * (b - a))

    # recompute actual extent with constant gap
    extent_um = sum(widths_um) + (N - 1) * g

    # Center the pattern inside the bounding box along the “array” axis
    axis_min_um = axis_min_dbu * dbu
    start_um = axis_min_um + 0.5 * (length_um - extent_um)

    # --- build stripes as a Region before clipping
    stripes = pya.Region()
    pos_um = start_um

    for i, w_um in enumerate(widths_um):
        w_dbu = um2dbu(w_um)

        if orientation == "vertical":
            x0 = um2dbu(pos_um)
            x1 = x0 + w_dbu
            stripe_box = pya.Box(x0, bbox.bottom, x1, bbox.top)
        else:
            y0 = um2dbu(pos_um)
            y1 = y0 + w_dbu
            stripe_box = pya.Box(bbox.left, y0, bbox.right, y1)

        stripes.insert(stripe_box)

        # advance by width + constant gap (except after last)
        pos_um += w_um
        if i < N - 1:
            pos_um += g

    # --- clip channels to target region and write to the layout
    channels_in_target = stripes & target_reg
    cell.shapes(layer_index).insert(channels_in_target)
    return channels_in_target


def _UM(layout, x_um):  # µm → dbu (int)
    return int(round(x_um / layout.dbu))

def _box_center(layout, xc_um, yc_um, lx_um, wy_um):
    x1 = _UM(layout, xc_um - 0.5*lx_um); y1 = _UM(layout, yc_um - 0.5*wy_um)
    x2 = _UM(layout, xc_um + 0.5*lx_um); y2 = _UM(layout, yc_um + 0.5*wy_um)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return db.Box(x1, y1, x2, y2)



def trapezoid_attach_vertical(layout, y_edge_um, x_center_um,
                              length_um, width_near_um, width_far_um,
                              side="up"):
    """
    Build a vertical trapezoid whose NEAR horizontal edge lies at y_edge_um
    and whose centerline is at x_center_um.  The trapezoid extends towards
    +Y ("up") or -Y ("down").  The near edge (touching the channel) has
    width_near_um; the far edge has width_far_um.  Returns pya.Polygon (dbu).
    """
    pya = db
    UM = lambda u: _UM(layout, u)

    assert length_um > 0 and width_near_um > 0 and width_far_um > 0

    y_near = y_edge_um
    y_far  = y_edge_um + length_um if side == "up" else y_edge_um - length_um
    x0     = x_center_um

    if side == "up":
        pts_um = [
            (x0 - 0.5*width_near_um, y_near),
            (x0 + 0.5*width_near_um, y_near),
            (x0 + 0.5*width_far_um,  y_far),
            (x0 - 0.5*width_far_um,  y_far),
        ]
    else:  # "down"
        pts_um = [
            (x0 - 0.5*width_far_um,  y_far),
            (x0 + 0.5*width_far_um,  y_far),
            (x0 + 0.5*width_near_um, y_near),
            (x0 - 0.5*width_near_um, y_near),
        ]

    pts_i = [pya.Point(UM(x), UM(y)) for (x, y) in pts_um]
    return pya.Polygon(pts_i)

import math

def add_star_marker(layout,
                    cell,
                    layer_index,
                    xc_um,
                    yc_um,
                    *,
                    arm_length_um=2600,
                    arm_width_um=200,
                    star_space_um=1600,
                    inner_length_um=50,
                    inner_width_um=50):
    """
    Draw one 'facility' marker composed of:
      - an 8-arm star (arms of length `arm_length_um`, width `arm_width_um`)
      - a small inner cross (size `inner_length_um` x `inner_width_um`)
    centered at (xc_um, yc_um) on `layer_index`.

    Relies on your existing Chamber(x_um, y_um, w_um, l_um, ... , angle_deg) helper,
    which inserts a rectangle centered at (x_um, y_um) rotated by angle.
    """
    pillar_size_um = 0,
    pillar_distance_um = 0
    angles_star  = (0, 45, 90, 135, 180, 225, 270, 315, 360)
    angles_cross = (0, 90, 180, 270, 360)

    # --- outer 8-arm star (sun)
    for ang in angles_star:
        a = math.radians(ang)
        x_rot = star_space_um * math.cos(a)
        y_rot = star_space_um * math.sin(a)
        # rotate the long axis along the arm direction -> Chamber angle = arm angle + 90
        cell.shapes(layer_index).insert(
            Chamber(xc_um + x_rot, yc_um + y_rot,
                    arm_width_um, arm_length_um,
                    0, pillar_size_um, pillar_distance_um,
                    ang + 90)
        )

    # --- inner cross
    for ang in angles_cross:
        a = math.radians(ang)
        x_rot = 0.5 * inner_length_um * math.cos(a)
        y_rot = 0.5 * inner_length_um * math.sin(a)
        cell.shapes(layer_index).insert(
            Chamber(xc_um + x_rot, yc_um + y_rot,
                    inner_width_um, inner_length_um,
                    0, pillar_size_um, pillar_distance_um,
                    ang + 90)
        )


def add_facility_markers(layout,
                         cell,
                         layer_index,
                         *,
                         # default: your three required marker positions
                         left_center=(-18000.0,     0.0),
                         right_top=  ( 18000.0,  10000.0),
                         right_mid=  ( 18000.0, -10000.0),
                         # geometry defaults (can override)
                         arm_length_um=2600,
                         arm_width_um=200,
                         star_space_um=1600,
                         inner_length_um=50,
                         inner_width_um=50):
    """
    Places the three standard microfluidics-facility markers on `layer_index`.
    All dimensions are tunable via kwargs.
    """
    for (xc, yc) in (left_center, right_top, right_mid):
        add_star_marker(layout, cell, layer_index, xc, yc,
                        arm_length_um=arm_length_um,
                        arm_width_um=arm_width_um,
                        star_space_um=star_space_um,
                        inner_length_um=inner_length_um,
                        inner_width_um=inner_width_um)

def build_air_gap_crossing(layout,
                           cell,
                           # layers
                           layer_ppa=1, layer_30=2, layer_mid=3,
                           # placement (device center)
                           x0_um=0.0, y0_um=0.0,
                           # middle (horizontal)
                           mid_length_um=2500.0,
                           mid_w_um=100.0,          # width (height) of each middle channel
                           n_middle=1,              # <-- NEW: number of parallel middle channels
                           mid_sep_um=150.0,        # <-- NEW: gap between middle channels (must be > mid_w_um)
                           # 30-µm buffers
                           buffer_len_um=120.0,
                           thin_w_um=30.0,
                           # side (vertical)
                           side_height_um=1800.0,
                           ppa_wall_w_um=300.0,
                           side_to_ppa_gap_um=0.0,  # 0 => share wall
                           # trapezoids (top & bottom per side)
                           trap_len_um=600.0,
                           trap_near_w_um=30.0,
                           trap_far_w_um=300.0,
                           # PPA grid
                           ppa_slit_w_um=4.0, ppa_spacing_um=50.0,
                           ppa_two_dirs=False, ppa_rotate_deg=0.0):
    """
    N parallel middle channels (layer_mid), each connected by 30 µm buffers (layer_30)
    to vertical 30 µm side channels (layer_30) that share a wall with tall PPA liners
    (layer_ppa). Each side channel also has top+bottom trapezoids (layer_30).
    """
    pya = db
    dbu = layout.dbu
    UM  = lambda u: int(round(u / dbu))

    # helper: centered box, l along X, w along Y
    def box_center(xc_um, yc_um, l_um, w_um):
        x1 = UM(xc_um - 0.5*l_um); y1 = UM(yc_um - 0.5*w_um)
        x2 = UM(xc_um + 0.5*l_um); y2 = UM(yc_um + 0.5*w_um)
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return pya.Box(x1, y1, x2, y2)

    # ---------- VALIDATION ----------
    if n_middle < 1:
        raise ValueError("n_middle must be ≥ 1")
    if mid_sep_um <= mid_w_um:
        raise ValueError(f"mid_sep_um ({mid_sep_um} µm) must be > mid_w_um ({mid_w_um} µm)")

    stack_h = n_middle*mid_w_um + (n_middle-1)*mid_sep_um  # total vertical envelope of middle stack
    if stack_h > side_height_um:
        # compute maximum feasible n
        max_n = int((side_height_um + mid_sep_um) // (mid_w_um + mid_sep_um))
        raise ValueError(
            f"Middle stack height {stack_h:.1f} µm exceeds side_height {side_height_um:.1f} µm. "
            f"Reduce n_middle or widths/spacings (max n_middle ≈ {max_n})."
        )

    # Precompute left/right x of all middle channels (same for each)
    mid_left_x_um  = x0_um - 0.5*mid_length_um
    mid_right_x_um = x0_um + 0.5*mid_length_um

    # ---------- 3) Vertical side channels (layer_30) ----------
    # Set by middle's left/right edges & buffer length; they must span the full stack height
    left_side_inner_x  = mid_left_x_um  - buffer_len_um
    right_side_inner_x = mid_right_x_um + buffer_len_um
    left_side_center_x  = left_side_inner_x  - 0.5*thin_w_um
    right_side_center_x = right_side_inner_x + 0.5*thin_w_um

    side_left_box  = box_center(left_side_center_x,  y0_um, thin_w_um, side_height_um)
    side_right_box = box_center(right_side_center_x, y0_um, thin_w_um, side_height_um)
    cell.shapes(layer_30).insert(side_left_box)
    cell.shapes(layer_30).insert(side_right_box)

    # ---------- 4) Tall PPA liners (layer_ppa) ----------
    left_ppa_center_x  = left_side_center_x  - 0.5*thin_w_um - 0.5*ppa_wall_w_um - side_to_ppa_gap_um
    right_ppa_center_x = right_side_center_x + 0.5*thin_w_um + 0.5*ppa_wall_w_um + side_to_ppa_gap_um
    ppa_left_box  = box_center(left_ppa_center_x,  y0_um, ppa_wall_w_um, side_height_um)
    ppa_right_box = box_center(right_ppa_center_x, y0_um, ppa_wall_w_um, side_height_um)

    fill_with_ppa_grid(layout, cell, ppa_left_box,  layer_ppa,
                       channel_width_um=ppa_slit_w_um, spacing_um=ppa_spacing_um,
                       two_directions=ppa_two_dirs, rotate_deg=ppa_rotate_deg)
    fill_with_ppa_grid(layout, cell, ppa_right_box, layer_ppa,
                       channel_width_um=ppa_slit_w_um, spacing_um=ppa_spacing_um,
                       two_directions=ppa_two_dirs, rotate_deg=ppa_rotate_deg)

    # ---------- 1+2) N middle channels (layer_mid) + their buffers (layer_30) ----------
    # Place them symmetrically about y0_um.
    mids_reg = pya.Region()
    bufs_reg = pya.Region()

    # first center y for the stack's top channel
    y_top_center = y0_um + 0.5*stack_h - 0.5*mid_w_um
    pitch = mid_w_um + mid_sep_um

    for i in range(n_middle):
        yc = y_top_center - i*pitch

        # middle
        mbox = box_center(x0_um, yc, mid_length_um, mid_w_um)
        cell.shapes(layer_mid).insert(mbox)
        mids_reg.insert(mbox)

        # buffers left/right at same yc
        left_buf_xc  = mid_left_x_um  - 0.5*buffer_len_um
        right_buf_xc = mid_right_x_um + 0.5*buffer_len_um
        bl = box_center(left_buf_xc,  yc, buffer_len_um, thin_w_um)
        br = box_center(right_buf_xc, yc, buffer_len_um, thin_w_um)
        cell.shapes(layer_30).insert(bl)
        cell.shapes(layer_30).insert(br)
        bufs_reg.insert(bl)
        bufs_reg.insert(br)

    # ---------- 5) Trapezoids (top & bottom per side) ----------
    pya = db
    left_top_y_um   = side_left_box.top    * dbu
    left_bot_y_um   = side_left_box.bottom * dbu
    right_top_y_um  = side_right_box.top   * dbu
    right_bot_y_um  = side_right_box.bottom* dbu
    left_xc_um      = (side_left_box.left  + side_left_box.right )*0.5*dbu
    right_xc_um     = (side_right_box.left + side_right_box.right)*0.5*dbu

    for (x_c, y_edge, updown) in [
        (left_xc_um,  left_top_y_um,  "up"),
        (left_xc_um,  left_bot_y_um,  "down"),
        (right_xc_um, right_top_y_um, "up"),
        (right_xc_um, right_bot_y_um, "down"),
    ]:
        trap_poly = trapezoid_attach_vertical(layout,
                        y_edge_um=y_edge, x_center_um=x_c,
                        length_um=trap_len_um,
                        width_near_um=trap_near_w_um, width_far_um=trap_far_w_um,
                        side=updown)
        cell.shapes(layer_30).insert(trap_poly)

    return {
        "mids": mids_reg,
        "buffers": bufs_reg,
        "side_channels": pya.Region(side_left_box) + pya.Region(side_right_box),
        "ppa_liners": pya.Region(ppa_left_box) + pya.Region(ppa_right_box),
    }

def build_air_gap_crossing_v2(
    layout,
    cell,
    # layers
    layer_ppa=1, layer_30=2, layer_mid=3,
    # placement (device center)
    x0_um=0.0, y0_um=0.0,
    # middle (horizontal)
    mid_length_um=2500.0,
    mid_w_um=100.0,          # width (height) of each middle channel
    n_middle=3,              # number of parallel middle channels
    mid_sep_um=150.0,        # gap between middle channels (> mid_w_um)
    # 30-µm buffers (left/right)
    buffer_len_um=120.0,
    thin_w_um=30.0,
    # side (vertical)
    side_height_um=1800.0,
    ppa_wall_w_um=300.0,
    side_to_ppa_gap_um=0.0,  # 0 => share wall
    # inlet/outlet funnels (TOP)
    trap_len_um=600.0,
    trap_near_w_um=30.0,
    trap_far_w_um=300.0,
    # PPA grid
    ppa_slit_w_um=4.0, ppa_spacing_um=50.0,
    ppa_two_dirs=False, ppa_rotate_deg=0.0,
    # center vertical riser
    center_vert_w_um=80.0,
    center_vert_overrun_um=0.0,
    # new funnel at bottom of center vertical channel
    center_trap_len_um=600.0,
    center_trap_near_w_um=30.0,
    center_trap_far_w_um=300.0,
    # --- NEW options ---
    # replace top-left/right + bottom-center trapezoids with circles on layer_ppa
    replace_trapezoids_with_circles=False,
    top_inlet_circle_d_um=750.0,         # 0.75 mm (top-left/right)
    bottom_center_circle_d_um=750.0,     # 0.75 mm (bottom-center)
    circle_sides=128,
    # side PPA circles (left/right of PPA blocks)
    add_ppa_side_circles=False,
    ppa_side_circle_d_um=750.0,          # 0.75 mm
    # extend side channels vertically
    side_extend_top_um=150.0,
    side_extend_bottom_um=150.0,
    # guaranteed overlaps at interfaces
    overlap_um=10.0
):
    """
    Adds:
      • Side PPA circles (left/right), centered at y0_um on layer_ppa.
      • Taller side channels via side_extend_top_um / side_extend_bottom_um.
      • Horizontal channels start at the new very bottom.
      • Tunable overlap between mids↔buffers and buffers↔side channels.
    """
    import math
    pya = db
    dbu = layout.dbu
    UM  = lambda u: int(round(u / dbu))

    def box_center(xc_um, yc_um, l_um, w_um):
        x1 = UM(xc_um - 0.5*l_um); y1 = UM(yc_um - 0.5*w_um)
        x2 = UM(xc_um + 0.5*l_um); y2 = UM(yc_um + 0.5*w_um)
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return pya.Box(x1, y1, x2, y2)

    def circle_polygon(xc_um, yc_um, d_um, nsides=circle_sides):
        r_um = 0.5*d_um
        pts = []
        ns = max(16, int(nsides))
        for k in range(ns):
            ang = 2.0*math.pi * (k / float(ns))
            x = xc_um + r_um*math.cos(ang)
            y = yc_um + r_um*math.sin(ang)
            pts.append(pya.Point(UM(x), UM(y)))
        return pya.Polygon(pts)

    if n_middle < 1:
        raise ValueError("n_middle must be ≥ 1")
    if mid_sep_um <= mid_w_um:
        raise ValueError(f"mid_sep_um ({mid_sep_um} µm) must be > mid_w_um ({mid_w_um} µm)")

    # Useful horizontals extents
    mid_left_x_um  = x0_um - 0.5*mid_length_um
    mid_right_x_um = x0_um + 0.5*mid_length_um

    # --- vertical side channels (layer_30), EXTENDED
    # Move the center so that the new top/bottom edges extend by the requested amounts
    eff_side_h = side_height_um + side_extend_top_um + side_extend_bottom_um
    ppa_side_h = side_height_um
    side_center_y = y0_um + 0.5*(side_extend_top_um - side_extend_bottom_um)

    left_side_inner_x  = mid_left_x_um  - buffer_len_um
    right_side_inner_x = mid_right_x_um + buffer_len_um
    left_side_center_x  = left_side_inner_x  - 0.5*thin_w_um
    right_side_center_x = right_side_inner_x + 0.5*thin_w_um

    side_left_box  = box_center(left_side_center_x,  side_center_y, thin_w_um, eff_side_h)
    side_right_box = box_center(right_side_center_x, side_center_y, thin_w_um, eff_side_h)
    cell.shapes(layer_30).insert(side_left_box)
    cell.shapes(layer_30).insert(side_right_box)

    side_top_y_um    = side_left_box.top    * dbu
    side_bottom_y_um = side_left_box.bottom * dbu

    # --- PPA liners
    left_ppa_center_x  = left_side_center_x  - 0.5*thin_w_um - 0.5*ppa_wall_w_um - side_to_ppa_gap_um
    right_ppa_center_x = right_side_center_x + 0.5*thin_w_um + 0.5*ppa_wall_w_um + side_to_ppa_gap_um
    ppa_left_box  = box_center(left_ppa_center_x,  side_center_y, ppa_wall_w_um, ppa_side_h)
    ppa_right_box = box_center(right_ppa_center_x, side_center_y, ppa_wall_w_um, ppa_side_h)

    fill_with_ppa_grid(layout, cell, ppa_left_box,  layer_ppa,
                       channel_width_um=ppa_slit_w_um, spacing_um=ppa_spacing_um,
                       two_directions=ppa_two_dirs, rotate_deg=ppa_rotate_deg)
    fill_with_ppa_grid(layout, cell, ppa_right_box, layer_ppa,
                       channel_width_um=ppa_slit_w_um, spacing_um=ppa_spacing_um,
                       two_directions=ppa_two_dirs, rotate_deg=ppa_rotate_deg)

    # --- OPTIONAL: side PPA circles (outside, centered at y0_um)
    ppa_side_circles_reg = pya.Region()
    if add_ppa_side_circles:
        r_side = 0.5*ppa_side_circle_d_um
        # Left PPA: circle to the OUTER left
        cL = circle_polygon(left_ppa_center_x - 0.5*ppa_wall_w_um - r_side, y0_um, ppa_side_circle_d_um)
        # Right PPA: circle to the OUTER right
        cR = circle_polygon(right_ppa_center_x + 0.5*ppa_wall_w_um + r_side, y0_um, ppa_side_circle_d_um)
        cell.shapes(layer_ppa).insert(cL); ppa_side_circles_reg.insert(cL)
        cell.shapes(layer_ppa).insert(cR); ppa_side_circles_reg.insert(cR)

    # --- middle channels + buffers
    # Place bottom channel so its bottom edge == side_bottom_y_um
    stack_h = n_middle*mid_w_um + (n_middle-1)*mid_sep_um
    if side_bottom_y_um + stack_h > side_top_y_um + 1e-9:
        max_n = int(( (side_top_y_um - side_bottom_y_um) + mid_sep_um ) // (mid_w_um + mid_sep_um))
        raise ValueError(
            f"Middle stack height {stack_h:.1f} µm exceeds available side height {side_top_y_um - side_bottom_y_um:.1f} µm "
            f"(max n_middle ≈ {max_n})."
        )

    mids_reg = pya.Region()
    bufs_reg = pya.Region()

    # Bottom-most channel center
    first_center_y = side_bottom_y_um + 0.5*mid_w_um
    pitch = mid_w_um + mid_sep_um

    # Effective drawn lengths to ensure overlaps:
    #  - mid boxes extend 'overlap_um' into buffers on both ends
    #  - buffers extend 'overlap_um' into the thin side channels
    eff_mid_len = mid_length_um + 2.0*overlap_um
    eff_buf_len = buffer_len_um + overlap_um

    for i in range(n_middle):
        yc = first_center_y + i*pitch

        # Middle channel (layer_mid) with overlap into buffers
        mbox = box_center(x0_um, yc, eff_mid_len, mid_w_um)
        cell.shapes(layer_mid).insert(mbox)
        mids_reg.insert(mbox)

        # Buffers (layer_30) overlapping side channels
        # Keep the inner edges fixed at mid_left_x_um / mid_right_x_um, extend outward by overlap_um
        left_buf_xc  = (mid_left_x_um - 0.5*buffer_len_um) - 0.5*overlap_um
        right_buf_xc = (mid_right_x_um + 0.5*buffer_len_um) + 0.5*overlap_um
        bl = box_center(left_buf_xc,  yc, eff_buf_len, thin_w_um)
        br = box_center(right_buf_xc, yc, eff_buf_len, thin_w_um)
        cell.shapes(layer_30).insert(bl); bufs_reg.insert(bl)
        cell.shapes(layer_30).insert(br); bufs_reg.insert(br)

    # --- center vertical riser (layer_mid)
    stack_top_edge_y = side_bottom_y_um + stack_h
    center_top_y     = stack_top_edge_y + center_vert_overrun_um
    center_height    = center_top_y - side_bottom_y_um
    center_vert_box  = box_center(x0_um, (center_top_y + side_bottom_y_um)/2.0,
                                  center_vert_w_um, center_height)
    cell.shapes(layer_mid).insert(center_vert_box)

    # --- trapezoids OR circle reservoirs (top-left/right + bottom-center)
    left_xc_um  = (side_left_box.left  + side_left_box.right )*0.5*dbu
    right_xc_um = (side_right_box.left + side_right_box.right)*0.5*dbu

    reservoir_circles = pya.Region()

    if replace_trapezoids_with_circles:
        # Top circles: centers just above the top edge
        r_top = 0.5*top_inlet_circle_d_um
        cTL = circle_polygon(left_xc_um,  side_top_y_um + r_top,  top_inlet_circle_d_um)
        cTR = circle_polygon(right_xc_um, side_top_y_um + r_top,  top_inlet_circle_d_um)
        cell.shapes(layer_ppa).insert(cTL); reservoir_circles.insert(cTL)
        cell.shapes(layer_ppa).insert(cTR); reservoir_circles.insert(cTR)

        # Bottom center circle: center just below the bottom edge
        r_bot = 0.5*bottom_center_circle_d_um
        cB = circle_polygon(x0_um, side_bottom_y_um - r_bot, bottom_center_circle_d_um)
        cell.shapes(layer_ppa).insert(cB); reservoir_circles.insert(cB)
    else:
        # Keep original trapezoids on layer_30
        for (x_c, y_edge) in [
            (left_xc_um,  side_top_y_um),
            (right_xc_um, side_top_y_um),
        ]:
            trap_poly = trapezoid_attach_vertical(
                layout,
                y_edge_um=y_edge, x_center_um=x_c,
                length_um=trap_len_um,
                width_near_um=trap_near_w_um, width_far_um=trap_far_w_um,
                side="up"
            )
            cell.shapes(layer_30).insert(trap_poly)

        trap_poly_bottom = trapezoid_attach_vertical(
            layout,
            y_edge_um=side_bottom_y_um,
            x_center_um=x0_um,
            length_um=center_trap_len_um,
            width_near_um=center_trap_near_w_um,
            width_far_um=center_trap_far_w_um,
            side="down"
        )
        cell.shapes(layer_30).insert(trap_poly_bottom)

    return {
        "mids": pya.Region(center_vert_box) + mids_reg,
        "buffers": bufs_reg,
        "side_channels": pya.Region(side_left_box) + pya.Region(side_right_box),
        "ppa_liners": pya.Region(ppa_left_box) + pya.Region(ppa_right_box),
        "reservoir_circles": reservoir_circles,
        "ppa_side_circles": ppa_side_circles_reg,
        "meta": {
            "effective_side_height_um": eff_side_h,
            "side_top_y_um": side_top_y_um,
            "side_bottom_y_um": side_bottom_y_um,
            "overlap_um": overlap_um
        }
    }

def _add_region_to_layer(cell, layer, region):

    # Use the SAME insertion path you used before (works with your Name(...))
    cell.shapes(layer).insert(region)




def build_air_gap_crossing_topPPA(
    layout,
    cell,
    # layers
    layer_ppa=1, layer_30=2, layer_mid=3,
    # placement (device center)
    x0_um=0.0, y0_um=0.0,
    # middle (horizontal)
    mid_length_um=2500.0,
    mid_w_um=100.0,
    n_middle=3,
    mid_sep_um=150.0,
    # buffers and side rails
    buffer_len_um=120.0,
    thin_w_um=30.0,          # side-rail width
    buffer_w_um=None,        # (1) NEW: buffer width; defaults to thin_w_um
    # side (vertical)
    side_height_um=1800.0,
    side_extend_top_um=150.0,
    side_extend_bottom_um=150.0,
    # PPA over the top extension band
    ppa_wall_w_um=300.0,
    ppa_slit_w_um=4.0, ppa_spacing_um=50.0,
    ppa_two_dirs=True, ppa_rotate_deg=0.0,
    # horizontal inlet/outlet (sits at the mid-stack Y; inner edge at rail centerline)
    hio_len_um=700.0,
    hio_w_um=None,           # (3) width of the bar that connects to the trapezoid
    # reservoirs at OUTER ends of the IO
    trap_len_um=600.0,
    trap_near_w_um=30.0,
    trap_far_w_um=300.0,
    replace_trapezoids_with_circles=False,
    top_inlet_circle_d_um=750.0,
    circle_sides=128,
    # center vertical riser + bottom funnel
    center_vert_w_um=80.0,
    center_vert_overrun_um=0.0,         # kept for backward compat (top only)
    center_vert_overrun_top_um=None,    # (2) NEW precise control
    center_vert_overrun_bottom_um=0.0,  # (2) NEW
    center_trap_len_um=600.0,
    center_trap_near_w_um=30.0,
    center_trap_far_w_um=300.0,
    # overlaps
    overlap_um=10.0,
    # (4) Optional bounding box
    draw_bbox=False,
    bbox_margin_um=200.0,
    layer_bbox=None,          # defaults to layer_30 if None
    # --- Optional chip label ---
    chip_name = None,  # e.g. "FranklinTopPPA v1". If None/"" => no text
    chip_text_layer = 1,  # layer to draw the text (you asked for layer 1)
    chip_text_h_um = 5000.0,  # text height [µm]
    chip_text_inset_x_um = 1000.0,  # inset from bbox left edge [µm]
    chip_text_inset_y_um = 1000.0,  # inset from bbox bottom edge [µm]
    bbox_channel_w_um=200.0,  # NEW: wall/channel thickness of the bounding frame
    chip_text_clearance_um=200.0,  # keep this much clearance from any device geometry
    chip_text_pad_x_um=300.0,      # extra width of the label plate beyond text bbox
    chip_text_pad_y_um=200.0,      # extra height of the label plate beyond text bbox
    chip_text_angle="R0",  # pass-through to your Name(...), e.g. "R0","R90","R180","R270"
    chip_text_size=100,  # size passed directly to Name(...), you control units/scale there

):
    """
    Differences vs your previous version:
      • buffer_w_um separates buffer width from thin side-rail width.
      • center_vert_overrun_top_um / bottom_um control extra height of the center riser beyond the stack.
      • hio_w_um already tunes the IO bar feeding the trapezoids (kept).
      • Optional bounding box with configurable margin and layer.
    """
    import math
    pya = db
    dbu = layout.dbu
    UM  = lambda u: int(round(u / dbu))

    def box_center(xc_um, yc_um, l_um, w_um):
        x1 = UM(xc_um - 0.5*l_um); y1 = UM(yc_um - 0.5*w_um)
        x2 = UM(xc_um + 0.5*l_um); y2 = UM(yc_um + 0.5*w_um)
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return pya.Box(x1, y1, x2, y2)

    def circle_polygon(xc_um, yc_um, d_um, nsides=circle_sides):
        r = 0.5*d_um; ns = max(16, int(nsides))
        pts = [pya.Point(UM(xc_um + r*math.cos(2*math.pi*k/ns)),
                         UM(yc_um + r*math.sin(2*math.pi*k/ns))) for k in range(ns)]
        return pya.Polygon(pts)

    def trapezoid_attach_horizontal(x_edge_um, y_center_um, length_um, w_near_um, w_far_um, side):
        s = -1.0 if side == "left" else 1.0
        x0, x1 = x_edge_um, x_edge_um + s*length_um
        wn, wf = 0.5*w_near_um, 0.5*w_far_um
        pts = [pya.Point(UM(x0), UM(y_center_um - wn)),
               pya.Point(UM(x0), UM(y_center_um + wn)),
               pya.Point(UM(x1), UM(y_center_um + wf)),
               pya.Point(UM(x1), UM(y_center_um - wf))]
        return pya.Polygon(pts)

    # ---- sanity
    if n_middle < 1: raise ValueError("n_middle must be ≥ 1")
    if mid_sep_um <= mid_w_um:
        raise ValueError(f"mid_sep_um ({mid_sep_um} µm) must be > mid_w_um ({mid_w_um} µm)")

    if buffer_w_um is None:
        buffer_w_um = thin_w_um
    if hio_w_um is None:
        hio_w_um = thin_w_um
    if layer_bbox is None:
        layer_bbox = layer_30

    # ---- convenient Region to compute bounding box at the end
    everything = pya.Region()

    # ---- horizontals & side rails
    mid_left_x_um  = x0_um - 0.5*mid_length_um
    mid_right_x_um = x0_um + 0.5*mid_length_um

    left_side_inner_x  = mid_left_x_um  - buffer_len_um
    right_side_inner_x = mid_right_x_um + buffer_len_um
    left_side_center_x  = left_side_inner_x  - 0.5*thin_w_um
    right_side_center_x = right_side_inner_x + 0.5*thin_w_um

    eff_side_h = side_height_um + side_extend_top_um + side_extend_bottom_um
    side_center_y = y0_um + 0.5*(side_extend_top_um - side_extend_bottom_um)

    side_left_box  = box_center(left_side_center_x,  side_center_y, thin_w_um, eff_side_h)
    side_right_box = box_center(right_side_center_x, side_center_y, thin_w_um, eff_side_h)
    cell.shapes(layer_30).insert(side_left_box);  everything.insert(side_left_box)
    cell.shapes(layer_30).insert(side_right_box); everything.insert(side_right_box)

    side_top_y_um    = side_left_box.top    * dbu
    side_bottom_y_um = side_left_box.bottom * dbu

    # ---- middle stack: bottom-aligned to rails (so no bottom margin)
    stack_h = n_middle*mid_w_um + (n_middle-1)*mid_sep_um
    avail_h = side_top_y_um - side_bottom_y_um
    if stack_h > avail_h + 1e-9:
        max_n = int((avail_h + mid_sep_um) // (mid_w_um + mid_sep_um))
        raise ValueError(f"Middle stack {stack_h:.1f} µm > available {avail_h:.1f} µm (max n≈{max_n}).")

    first_center_y = side_bottom_y_um + 0.5*mid_w_um
    pitch = mid_w_um + mid_sep_um

    eff_mid_len = mid_length_um + 2.0*overlap_um
    eff_buf_len = buffer_len_um + overlap_um
    left_buf_xc  = (mid_left_x_um - 0.5*buffer_len_um) - 0.5*overlap_um
    right_buf_xc = (mid_right_x_um + 0.5*buffer_len_um) + 0.5*overlap_um

    for i in range(n_middle):
        yc = first_center_y + i*pitch
        mbox = box_center(x0_um, yc, eff_mid_len, mid_w_um)
        cell.shapes(layer_mid).insert(mbox); everything.insert(mbox)

        bl = box_center(left_buf_xc,  yc, eff_buf_len, buffer_w_um)
        br = box_center(right_buf_xc, yc, eff_buf_len, buffer_w_um)
        cell.shapes(layer_30).insert(bl); everything.insert(bl)
        cell.shapes(layer_30).insert(br); everything.insert(br)

    # mid-stack centerline Y (between upper/lower halves) — where we run horizontal IO
    y_mid_stack = first_center_y + 0.5*stack_h

    # ---- center vertical riser (with overruns) + bottom funnel
    # support old single-parameter overrun (top only) or new top/bottom pair
    top_overrun = center_vert_overrun_top_um if (center_vert_overrun_top_um is not None) else max(0.0, center_vert_overrun_um)
    bot_overrun = max(0.0, center_vert_overrun_bottom_um)

    top_of_stack = first_center_y + (n_middle-1)*pitch + 0.5*mid_w_um
    center_top_y    = top_of_stack + top_overrun
    center_bottom_y = side_bottom_y_um - bot_overrun
    center_vert_h   = center_top_y - center_bottom_y
    center_vert_box = box_center(x0_um, 0.5*(center_top_y + center_bottom_y),
                                 center_vert_w_um, center_vert_h)
    cell.shapes(layer_mid).insert(center_vert_box); everything.insert(center_vert_box)

    trap_poly_bottom = trapezoid_attach_vertical(
        layout,
        y_edge_um=center_bottom_y,
        x_center_um=x0_um,
        length_um=center_trap_len_um,
        width_near_um=center_trap_near_w_um,
        width_far_um=center_trap_far_w_um,
        side="down"
    )
    cell.shapes(layer_30).insert(trap_poly_bottom); everything.insert(trap_poly_bottom)

    # ---- PPA only over the top extension band of the side rails
    ppa_band_h_um   = side_extend_top_um
    ppa_band_center = side_top_y_um - 0.5*side_extend_top_um

    ppa_left  = box_center(left_side_center_x,  ppa_band_center, ppa_wall_w_um, ppa_band_h_um)
    ppa_right = box_center(right_side_center_x, ppa_band_center, ppa_wall_w_um, ppa_band_h_um)

    fill_with_ppa_grid(layout, cell, ppa_left,  layer_ppa,
                       channel_width_um=ppa_slit_w_um, spacing_um=ppa_spacing_um,
                       two_directions=ppa_two_dirs, rotate_deg=ppa_rotate_deg)
    fill_with_ppa_grid(layout, cell, ppa_right, layer_ppa,
                       channel_width_um=ppa_slit_w_um, spacing_um=ppa_spacing_um,
                       two_directions=ppa_two_dirs, rotate_deg=ppa_rotate_deg)
    everything.insert(ppa_left); everything.insert(ppa_right)

    # ---- Horizontal IO at mid-stack Y; inner edge at rail centerline
    left_io  = box_center(left_side_center_x  - 0.5*hio_len_um,  y_mid_stack, hio_len_um, hio_w_um)
    right_io = box_center(right_side_center_x + 0.5*hio_len_um, y_mid_stack, hio_len_um, hio_w_um)
    cell.shapes(layer_30).insert(left_io);  everything.insert(left_io)
    cell.shapes(layer_30).insert(right_io); everything.insert(right_io)

    if replace_trapezoids_with_circles:
        r = 0.5*top_inlet_circle_d_um
        cL = circle_polygon(left_io.left*dbu - r,  y_mid_stack, top_inlet_circle_d_um)
        cR = circle_polygon(right_io.right*dbu + r, y_mid_stack, top_inlet_circle_d_um)
        cell.shapes(layer_ppa).insert(cL); everything.insert(cL)
        cell.shapes(layer_ppa).insert(cR); everything.insert(cR)
    else:
        trapL = trapezoid_attach_horizontal(left_io.left*dbu,  y_mid_stack,
                                            trap_len_um, trap_near_w_um, trap_far_w_um, "left")
        trapR = trapezoid_attach_horizontal(right_io.right*dbu, y_mid_stack,
                                            trap_len_um, trap_near_w_um, trap_far_w_um, "right")
        cell.shapes(layer_30).insert(trapL); everything.insert(trapL)
        cell.shapes(layer_30).insert(trapR); everything.insert(trapR)

    # ---- Optional bounding box with margin (expand-compatible)
    # ---- Optional bounding box with margin (expand-compatible) + optional chip label
    # ---- Optional bounding frame (ring) + optional chip label
    if not everything.is_empty():
        bb = everything.bbox()  # in dbu
        m = UM(bbox_margin_um) if 'bbox_margin_um' in locals() else 0
        if draw_bbox:
            # Outer and inner boxes
            outer = pya.Box(bb.left - m, bb.bottom - m, bb.right + m, bb.top + m)
            t = UM(bbox_channel_w_um)
            inner = pya.Box(outer.left + t, outer.bottom + t, outer.right - t, outer.top - t)

            # Draw as a ring: outer minus inner
            ring = pya.Region(outer) - pya.Region(inner)
            tgt_layer = layer_bbox if layer_bbox is not None else layer_30
            cell.shapes(tgt_layer).insert(ring)

        # --- chip name (direct polygon text) well inside the frame opening
        # --- chip name (polygon text) placed directly, no plate, no knock-out ---
        # --- chip name (use user's Name(...) to build a Region, then add to chosen layer)
        if chip_name:
            # Compute the frame opening in µm (so we can pass µm to Name)
            bb = everything.bbox()  # dbu
            outer_left_um = bb.left * dbu - bbox_margin_um
            outer_bottom_um = bb.bottom * dbu - bbox_margin_um
            outer_right_um = bb.right * dbu + bbox_margin_um
            outer_top_um = bb.top * dbu + bbox_margin_um

            t_ring_um = bbox_channel_w_um if draw_bbox else 0.0

            inner_left_um = outer_left_um + t_ring_um
            inner_bottom_um = outer_bottom_um + t_ring_um
            inner_right_um = outer_right_um - t_ring_um
            inner_top_um = outer_top_um - t_ring_um

            # Anchor inside the opening (bottom-left) with your insets
            tx_um = inner_left_um + chip_text_inset_x_um
            ty_um = inner_bottom_um + chip_text_inset_y_um

            # Build the text as a Region using YOUR function (which "works" in your env)
            txt_region = Name(
                xcor=tx_um,
                ycor=ty_um,
                text=str(chip_name),
                angle=chip_text_angle,
                size=chip_text_size,  # you control this scaling inside Name(...)
            )

            # Drop the Region onto the target layer as real polygons
            _add_region_to_layer(cell, chip_text_layer, txt_region)

    # ---- return
    return {
        "meta": {
            "side_top_y_um": side_top_y_um,
            "side_bottom_y_um": side_bottom_y_um,
            "y_mid_stack_um": y_mid_stack,
            "center_top_y_um": center_top_y,
            "center_bottom_y_um": center_bottom_y,
            "overlap_um": overlap_um
        }
    }

def build_air_gap_T_simple(
    layout,
    cell,
    # layers
    layer_channel=2,
    layer_ppa=1,
    # placement
    x0_um=0.0, y0_um=0.0,

    # central middle bar
    mid_len_um=1200.0,
    mid_w_um=200.0,

    # thin rails
    rail_w_um=40.0,
    rail_left_len_um=2000.0,
    rail_right_len_um=2000.0,

    # NEW: left-of-left extension (horizontal)
    left_ext_enable=True,         # turn the extra channel on/off
    left_ext_len_um=1500.0,       # length along x (to the left)
    left_ext_w_um=80.0,           # width along y
    left_ext_gap_um=0.0,          # optional gap between left rail and extension (0 = connected)

    # PPA over rails (unchanged)
    ppa_over_rail="both",         # "left", "right", "both", "none"
    ppa_band_w_um=1000.0,
    ppa_band_h_um=800.0,
    ppa_gap_y_um=0.0,
    ppa_slit_w_um=4.0, ppa_spacing_um=50.0,
    ppa_two_dirs=True, ppa_rotate_deg=0.0,

    # vertical riser + funnel
    vert_w_um=200.0,
    vert_down_len_um=1500.0,
    add_bottom_funnel=True,
    funnel_len_um=400.0,
    funnel_near_w_um=80.0,
    funnel_far_w_um=300.0,

    overlap_um=4.0,
    chip_name=None, chip_text_layer=1, chip_text_angle="R0", chip_text_size=100,
    chip_text_inset_x_um=1000.0, chip_text_inset_y_um=1000.0,
    draw_bbox=False, bbox_margin_um=1500.0, layer_bbox=None,
):
    pya = db
    dbu = layout.dbu
    UM  = lambda u: int(round(u / dbu))

    def box_center(xc_um, yc_um, l_um, w_um):
        x1 = UM(xc_um - 0.5*l_um); y1 = UM(yc_um - 0.5*w_um)
        x2 = UM(xc_um + 0.5*l_um); y2 = UM(yc_um + 0.5*w_um)
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return pya.Box(x1, y1, x2, y2)

    def trapezoid_attach_vertical(y_edge_um, x_center_um, length_um, width_near_um, width_far_um, side="down"):
        s = -1.0 if side == "up" else 1.0
        y0, y1 = y_edge_um, y_edge_um + s*length_um
        wn, wf = 0.5*width_near_um, 0.5*width_far_um
        pts = [
            pya.Point(UM(x_center_um - wn), UM(y0)),
            pya.Point(UM(x_center_um + wn), UM(y0)),
            pya.Point(UM(x_center_um + wf), UM(y1)),
            pya.Point(UM(x_center_um - wf), UM(y1)),
        ]
        return pya.Polygon(pts)

    if layer_bbox is None:
        layer_bbox = layer_channel

    everything = pya.Region()

    # --- middle bar
    mid_box = box_center(x0_um, y0_um, mid_len_um + 2*overlap_um, mid_w_um)
    cell.shapes(layer_channel).insert(mid_box); everything.insert(mid_box)
    mid_left_x  = x0_um - 0.5*mid_len_um
    mid_right_x = x0_um + 0.5*mid_len_um

    # --- rails
    left_rail = right_rail = None
    if rail_left_len_um > 0:
        left_center_x = mid_left_x - 0.5*rail_left_len_um
        left_rail = box_center(left_center_x, y0_um, rail_left_len_um + overlap_um, rail_w_um)
        cell.shapes(layer_channel).insert(left_rail); everything.insert(left_rail)
    if rail_right_len_um > 0:
        right_center_x = mid_right_x + 0.5*rail_right_len_um
        right_rail = box_center(right_center_x, y0_um, rail_right_len_um + overlap_um, rail_w_um)
        cell.shapes(layer_channel).insert(right_rail); everything.insert(right_rail)

    # --- NEW: left-of-left extension (horizontal, co-linear)
    # outer end of left rail:
    left_rail_outer_x = mid_left_x - rail_left_len_um
    if left_ext_enable and left_ext_len_um > 0 and left_ext_w_um > 0:
        ext_center_x = left_rail_outer_x - left_ext_gap_um - 0.5*left_ext_len_um
        ext_box = box_center(ext_center_x, y0_um, left_ext_len_um + (0 if left_ext_gap_um>0 else overlap_um), left_ext_w_um)
        cell.shapes(layer_channel).insert(ext_box); everything.insert(ext_box)

    # --- PPA over rails (unchanged; centered on *rails*, not including extension)
    def place_ppa_over_rail(rail_len_um: float, rail_center_x: float):
        band_w = min(ppa_band_w_um, max(rail_len_um, 0.0))
        if band_w <= 0 or ppa_band_h_um <= 0:
            return
        band_center_y = y0_um + 0.5*rail_w_um + 0.5*ppa_band_h_um + ppa_gap_y_um
        ppa_box = box_center(rail_center_x, band_center_y, band_w, ppa_band_h_um)
        # cell.shapes(layer_ppa).insert(ppa_box);
        everything.insert(ppa_box)
        fill_with_ppa_grid(layout, cell, ppa_box, layer_ppa,
                           channel_width_um=ppa_slit_w_um, spacing_um=ppa_spacing_um,
                           two_directions=ppa_two_dirs, rotate_deg=ppa_rotate_deg)

    if ppa_over_rail in ("left","both") and left_rail is not None:
        place_ppa_over_rail(rail_left_len_um, mid_left_x - 0.5*rail_left_len_um)
    if ppa_over_rail in ("right","both") and right_rail is not None:
        place_ppa_over_rail(rail_right_len_um, mid_right_x + 0.5*rail_right_len_um)

    # --- vertical riser + funnel
    if vert_down_len_um > 0:
        riser_center_y = y0_um - 0.5*vert_down_len_um
        riser = box_center(x0_um, riser_center_y, vert_w_um, vert_down_len_um + overlap_um)
        cell.shapes(layer_channel).insert(riser); everything.insert(riser)
        if add_bottom_funnel and funnel_len_um > 0:
            y_edge = riser.bottom * dbu
            trap = trapezoid_attach_vertical(
                y_edge_um=y_edge, x_center_um=x0_um,
                length_um=funnel_len_um,
                width_near_um=funnel_near_w_um, width_far_w_um=funnel_far_w_um if False else funnel_far_w_um,
                side="down"
            )
            cell.shapes(layer_channel).insert(trap); everything.insert(trap)

    if not everything.is_empty():
        bb = everything.bbox()  # in dbu
        m = UM(bbox_margin_um) if 'bbox_margin_um' in locals() else 0
        if draw_bbox:
            # Outer and inner boxes
            outer = pya.Box(bb.left - m, bb.bottom - m, bb.right + m, bb.top + m)
            t = UM(bbox_channel_w_um)
            inner = pya.Box(outer.left + t, outer.bottom + t, outer.right - t, outer.top - t)

            # Draw as a ring: outer minus inner
            ring = pya.Region(outer) - pya.Region(inner)
            tgt_layer = layer_bbox if layer_bbox is not None else layer_30
            cell.shapes(tgt_layer).insert(ring)

        # --- chip name (direct polygon text) well inside the frame opening
        # --- chip name (polygon text) placed directly, no plate, no knock-out ---
        # --- chip name (use user's Name(...) to build a Region, then add to chosen layer)
        if chip_name:
            # Compute the frame opening in µm (so we can pass µm to Name)
            bb = everything.bbox()  # dbu
            outer_left_um = bb.left * dbu - bbox_margin_um
            outer_bottom_um = bb.bottom * dbu - bbox_margin_um
            outer_right_um = bb.right * dbu + bbox_margin_um
            outer_top_um = bb.top * dbu + bbox_margin_um

            t_ring_um = bbox_channel_w_um if draw_bbox else 0.0

            inner_left_um = outer_left_um + t_ring_um
            inner_bottom_um = outer_bottom_um + t_ring_um
            inner_right_um = outer_right_um - t_ring_um
            inner_top_um = outer_top_um - t_ring_um

            # Anchor inside the opening (bottom-left) with your insets
            tx_um = inner_left_um + chip_text_inset_x_um
            ty_um = inner_bottom_um + chip_text_inset_y_um

            # Build the text as a Region using YOUR function (which "works" in your env)
            txt_region = Name(
                xcor=tx_um,
                ycor=ty_um,
                text=str(chip_name),
                angle=chip_text_angle,
                size=chip_text_size,  # you control this scaling inside Name(...)
            )

            # Drop the Region onto the target layer as real polygons
            _add_region_to_layer(cell, chip_text_layer, txt_region)
    return {
        "mid_left_x_um": mid_left_x,
        "mid_right_x_um": mid_right_x,
        "left_rail_outer_x_um": left_rail_outer_x,
        "left_extension_outer_x_um": (left_rail_outer_x - left_ext_gap_um - left_ext_len_um) if left_ext_enable else None,
    }

def fill_ppa_pair_in_rect(layout,
                                            cell,
                                            layer_index,
                                            rect_width_um,
                                            rect_height_um,
                                            main_channel_width_um,
                                            origin_x_um=0.0,
                                            origin_y_um=0.0,
                                            channel_orientation="horizontal",
                                            ppa_channel_width_um=4.0,
                                            ppa_spacing_um=50.0,
                                            ppa_two_directions=True,
                                            ppa_rotate_deg=0):
    """
    Create TWO PPA plates inside a rectangle, separated by a central
    straight main channel that is ALSO DRAWN as a solid polygon.

    Outer rectangle in microns:
        X in [origin_x_um, origin_x_um + rect_width_um]
        Y in [origin_y_um, origin_y_um + rect_height_um]

    Parameters
    ----------
    layout : pya.Layout
    cell   : pya.Cell
    layer_index : int
        Layer where *both* the PPA grid channels and the main channel
        polygon will be drawn (negative mold: channels are positive).
    rect_width_um  : float
        Total width of the rectangle (µm)
    rect_height_um : float
        Total height of the rectangle (µm)
    main_channel_width_um : float
        Width of the central straight channel (µm)
    origin_x_um, origin_y_um : float
        Lower-left corner of the rectangle (µm)
    channel_orientation : {"horizontal", "vertical"}
        "horizontal" -> main channel runs along X, plates above/below
        "vertical"   -> main channel runs along Y, plates left/right
    ppa_channel_width_um : float
        Width of individual PPA channels (passed to fill_with_ppa_grid)
    ppa_spacing_um : float
        Spacing between PPA channels (passed to fill_with_ppa_grid)
    ppa_two_directions : bool
        Whether PPA grid is 2D or 1D (passed to fill_with_ppa_grid)
    ppa_rotate_deg : float
        Rotation of the PPA grid (passed to fill_with_ppa_grid)

    Returns
    -------
    (ppa_reg_1, ppa_reg_2, main_channel_reg) : tuple of pya.Region
        ppa_reg_1, ppa_reg_2 : regions inserted for each PPA plate
        main_channel_reg     : region inserted for the central channel polygon
    """
    pya = db
    dbu = layout.dbu

    def um2dbu(x: float) -> int:
        return int(round(x / dbu))

    # --- 1) Build outer rectangle as Box
    x0 = um2dbu(origin_x_um)
    y0 = um2dbu(origin_y_um)
    x1 = um2dbu(origin_x_um + rect_width_um)
    y1 = um2dbu(origin_y_um + rect_height_um)

    outer_box = pya.Box(x0, y0, x1, y1)
    outer_reg = pya.Region(outer_box)
    if outer_reg.is_empty():
        return pya.Region(), pya.Region(), pya.Region()

    bbox = outer_reg.bbox()
    chan_w_dbu = um2dbu(main_channel_width_um)

    # --- 2) Define plates + main channel rectangles in dbu
    orientation = (channel_orientation or "horizontal").lower()

    if orientation.startswith("h"):
        # Main channel horizontal: width in Y
        mid_y = (bbox.bottom + bbox.top) // 2
        y0c = mid_y - chan_w_dbu // 2
        y1c = mid_y + chan_w_dbu // 2

        # Explicit Box for main channel
        main_chan_box = pya.Box(bbox.left, y0c, bbox.right, y1c)

        # Two plate boxes
        plate_top_box    = pya.Box(bbox.left, y1c,         bbox.right, bbox.top)
        plate_bottom_box = pya.Box(bbox.left, bbox.bottom, bbox.right, y0c)

        plate_reg_1 = outer_reg & pya.Region(plate_top_box)      # top plate
        plate_reg_2 = outer_reg & pya.Region(plate_bottom_box)   # bottom plate

    else:
        # Main channel vertical: width in X
        mid_x = (bbox.left + bbox.right) // 2
        x0c = mid_x - chan_w_dbu // 2
        x1c = mid_x + chan_w_dbu // 2

        # Explicit Box for main channel
        main_chan_box = pya.Box(x0c, bbox.bottom, x1c, bbox.top)

        # Two plate boxes
        plate_left_box  = pya.Box(bbox.left,  bbox.bottom, x0c,       bbox.top)
        plate_right_box = pya.Box(x1c,        bbox.bottom, bbox.right, bbox.top)

        plate_reg_1 = outer_reg & pya.Region(plate_left_box)    # left plate
        plate_reg_2 = outer_reg & pya.Region(plate_right_box)   # right plate

    # --- 3) Actually draw the main channel as a BOX shape
    # This is the key bit: this *is* a polygon on layer_index.
    cell.shapes(layer_index).insert(main_chan_box)
    main_channel_reg = pya.Region(main_chan_box)

    # --- 4) Fill the two plate regions with your PPA grid
    ppa_reg_1 = fill_with_ppa_grid(layout, cell, plate_reg_1, layer_index,
                                   channel_width_um=ppa_channel_width_um,
                                   spacing_um=ppa_spacing_um,
                                   two_directions=ppa_two_directions,
                                   rotate_deg=ppa_rotate_deg)

    ppa_reg_2 = fill_with_ppa_grid(layout, cell, plate_reg_2, layer_index,
                                   channel_width_um=ppa_channel_width_um,
                                   spacing_um=ppa_spacing_um,
                                   two_directions=ppa_two_directions,
                                   rotate_deg=ppa_rotate_deg)

    return ppa_reg_1, ppa_reg_2, main_channel_reg


def fill_two_ppa_with_single_channel(layout,
                                       cell,
                                       layer_index,
                                       strip_length_um,
                                       rect1_width_um,
                                       rect2_width_um,
                                       main_channel_width_um,
                                       side_channel_width_um,
                                       origin_x_um=0.0,
                                       origin_y_um=0.0,
                                       orientation="horizontal",
                                       ppa1_channel_width_um=4.0,
                                       ppa1_spacing_um=50.0,
                                       ppa2_channel_width_um=4.0,
                                       ppa2_spacing_um=50.0,
                                       ppa3_channel_width_um=4.0,
                                       ppa3_spacing_um=50.0,
                                       two_directions=True,
                                       rotate_deg=0):
    """
    Build a strip with THREE PPA regions and ONE large straight channel:

        PPA1-block | PPA2-block | MAIN CHANNEL | PPA3-block

    Each PPA-block has its own PPA channel width, spacing, and macro width.

    Geometry (in microns):
        - strip_length_um : length along the LARGE channel direction.
        - Along the stacking axis (perpendicular to the large channel), we have:
              rect1_width_um  (PPA1 block)
            + rect2_width_um  (PPA2 block)
            + main_channel_width_um (large channel)
            + rect3_width_um  (PPA3 block)

    Orientation
    -----------
    orientation = "horizontal":
        - Large channel runs along X (strip_length_um in X).
        - Blocks are stacked in Y from bottom to top:
            [PPA1][PPA2][MAIN CHANNEL][PPA3]

    orientation = "vertical":
        - Large channel runs along Y (strip_length_um in Y).
        - Blocks are stacked in X from left to right:
            [PPA1][PPA2][MAIN CHANNEL][PPA3]

    Parameters
    ----------
    layout : pya.Layout
    cell   : pya.Cell
    layer_index : int
        Layer where both the PPA grids and the large channel polygon are drawn.
    strip_length_um : float
        Length of the strip along the large channel direction.
    rect1_width_um, rect2_width_um, rect3_width_um : float
        Macro widths of PPA1, PPA2, PPA3 blocks along the stacking axis.
    main_channel_width_um : float
        Width of the single large straight channel between PPA2 and PPA3.
    origin_x_um, origin_y_um : float
        Lower-left corner of the strip (for horizontal) or
        bottom-left corner (for vertical), i.e. starting point.
    ppa1_channel_width_um, ppa1_spacing_um : float
        Micro-channel width and spacing for PPA1.
    ppa2_channel_width_um, ppa2_spacing_um : float
        Micro-channel width and spacing for PPA2.
    ppa3_channel_width_um, ppa3_spacing_um : float
        Micro-channel width and spacing for PPA3.
    two_directions, rotate_deg :
        Passed through to fill_with_ppa_grid for all three PPAs.

    Returns
    -------
    (ppa1_reg, ppa2_reg, ppa3_reg, main_channel_reg) : tuple of pya.Region
        ppa1_reg        : PPA polygons inserted for PPA1 block
        ppa2_reg        : PPA polygons inserted for PPA2 block
        ppa3_reg        : PPA polygons inserted for PPA3 block
        main_channel_reg: Region of the single large straight channel polygon
    """
    pya = db
    dbu = layout.dbu

    def um2dbu(x: float) -> int:
        return int(round(x / dbu))

    # Convert sizes to database units
    ppa2_channel_width_um_dbu = um2dbu(ppa2_channel_width_um)
    strip_len_dbu = um2dbu(strip_length_um)
    w1_dbu = um2dbu(rect1_width_um)
    w2_dbu = um2dbu(rect2_width_um)
    chan_dbu = um2dbu(main_channel_width_um)
    chan_side_dbu = um2dbu(side_channel_width_um)


    ox = um2dbu(origin_x_um)
    oy = um2dbu(origin_y_um)

    orientation = (orientation or "horizontal").lower()

    # --- 1) Define the four boxes: PPA1, PPA2, MAIN CHANNEL, PPA3
    if orientation.startswith("h"):
        # Large channel along X, stack along Y
        x0 = ox
        x1 = ox + strip_len_dbu

        y = oy
        # PPA1 block
        box1 = pya.Box(x0, y, x1, y + w1_dbu)
        y += w1_dbu
        # connecting channel
        ch_box_connect = pya.Box(x0, y, x1, y + ppa2_channel_width_um_dbu)
        y += ppa2_channel_width_um_dbu
        # PPA2 block
        box2 = pya.Box(x0, y, x1, y + w2_dbu)
        y += w2_dbu

        # Large main channel
        ch_box = pya.Box(x0, y, x1, y + chan_dbu)
        y += chan_dbu

        # PPA3 block
        box3 = pya.Box(x0, y, x1, y + w2_dbu)
        y += w2_dbu

        ch_box_connect2 = pya.Box(x0, y, x1, y + ppa2_channel_width_um_dbu)
        y += ppa2_channel_width_um_dbu
        # PPA2 block
        box4 = pya.Box(x0, y, x1, y + w1_dbu)
        y += w1_dbu
        ch_box2 = pya.Box(x0, y, x1, y + chan_side_dbu)
        y += chan_side_dbu

    else:
        # Large channel along Y, stack along X
        y0 = oy
        y1 = oy + strip_len_dbu

        x = ox
        # Correct version:
        box1 = pya.Box(x, y0, x + w1_dbu, y1)
        x += w1_dbu
        # connecting channel
        ch_box_connect = pya.Box(x, y0, x+ppa2_channel_width_um_dbu, y1)
        x += ppa2_channel_width_um_dbu
        # PPA2 block
        box2 = pya.Box(x, y0, x + w2_dbu, y1)
        x += w2_dbu

        # Large main channel
        ch_box = pya.Box(x, y0, x + chan_dbu, y1)
        x += chan_dbu

        # PPA3 block
        box3 = pya.Box(x, y0, x + w2_dbu, y1)
        x += w2_dbu

        ch_box_connect2 = pya.Box(x, y0, x+ppa2_channel_width_um_dbu, y1)
        x += ppa2_channel_width_um_dbu
        # PPA2 block
        box4 = pya.Box(x, y0, x + w1_dbu, y1)
        x += w2_dbu
        ch_box2 = pya.Box(x, y0, x + chan_side_dbu, y1)
        x += chan_side_dbu

    # --- 2) Draw the single large channel polygon explicitly
    cell.shapes(layer_index).insert(ch_box)
    main_channel_reg = pya.Region(ch_box)
    cell.shapes(layer_index).insert(ch_box2)
    main_channel_reg2 = pya.Region(ch_box2)
    cell.shapes(layer_index).insert(ch_box_connect)
    connect_channel_reg = pya.Region(ch_box_connect)
    cell.shapes(layer_index).insert(ch_box_connect2)
    connect_channel_reg = pya.Region(ch_box_connect2)
    # --- 3) Fill each PPA box using your existing PPA function

    # PPA1
    ppa1_reg = fill_with_ppa_grid(layout, cell, box1, layer_index,
                                  channel_width_um=ppa1_channel_width_um,
                                  spacing_um=ppa1_spacing_um,
                                  two_directions=two_directions,
                                  rotate_deg=rotate_deg)

    # PPA2
    ppa2_reg = fill_with_variable_channels(layout, cell, box2, layer_index,
                                    min_width_um = 10,
                                    max_width_um = 50,
                                    orientation = "vertical")
    ppa3_reg = fill_with_variable_channels(layout, cell, box3, layer_index,
                                    min_width_um = 10,
                                    max_width_um = 50,
                                    orientation = "vertical")
    # PPA3
    ppa4_reg = fill_with_ppa_grid(layout, cell, box4, layer_index,
                                  channel_width_um=ppa3_channel_width_um,
                                  spacing_um=ppa3_spacing_um,
                                  two_directions=two_directions,
                                  rotate_deg=rotate_deg)

    return ppa1_reg, ppa2_reg, ppa3_reg, main_channel_reg,connect_channel_reg


def final_design(layout,
                                       cell,
                                       layer_index,
                                       strip_length_um,
                                       rect1_width_um,
                                       rect2_width_um,
                                       main_channel_width_um,
                                       side_channel_width_um,
                                       origin_x_um=0.0,
                                       origin_y_um=0.0,
                                       orientation="horizontal",
                                       ppa1_channel_width_um=4.0,
                                       ppa1_spacing_um=50.0,
                                       ppa2_channel_width_um=4.0,
                                       ppa2_spacing_um=50.0,
                                       ppa3_channel_width_um=4.0,
                                       ppa3_spacing_um=50.0,
                                       channel_extend_left_um=1000.0,
                                       channel_extend_right_um=1500.0,
                                       channel_split_ratio=0.5,
                                       two_directions=True,
                                       rotate_deg=0):
    """
    Build a strip with THREE PPA regions and ONE large straight channel:

        PPA1-block | PPA2-block | MAIN CHANNEL | PPA3-block

    Each PPA-block has its own PPA channel width, spacing, and macro width.

    Geometry (in microns):
        - strip_length_um : length along the LARGE channel direction.
        - Along the stacking axis (perpendicular to the large channel), we have:
              rect1_width_um  (PPA1 block)
            + rect2_width_um  (PPA2 block)
            + main_channel_width_um (large channel)
            + rect3_width_um  (PPA3 block)

    Orientation
    -----------
    orientation = "horizontal":
        - Large channel runs along X (strip_length_um in X).
        - Blocks are stacked in Y from bottom to top:
            [PPA1][PPA2][MAIN CHANNEL][PPA3]

    orientation = "vertical":
        - Large channel runs along Y (strip_length_um in Y).
        - Blocks are stacked in X from left to right:
            [PPA1][PPA2][MAIN CHANNEL][PPA3]

    Parameters
    ----------
    layout : pya.Layout
    cell   : pya.Cell
    layer_index : int
        Layer where both the PPA grids and the large channel polygon are drawn.
    strip_length_um : float
        Length of the strip along the large channel direction.
    rect1_width_um, rect2_width_um, rect3_width_um : float
        Macro widths of PPA1, PPA2, PPA3 blocks along the stacking axis.
    main_channel_width_um : float
        Width of the single large straight channel between PPA2 and PPA3.
    origin_x_um, origin_y_um : float
        Lower-left corner of the strip (for horizontal) or
        bottom-left corner (for vertical), i.e. starting point.
    ppa1_channel_width_um, ppa1_spacing_um : float
        Micro-channel width and spacing for PPA1.
    ppa2_channel_width_um, ppa2_spacing_um : float
        Micro-channel width and spacing for PPA2.
    ppa3_channel_width_um, ppa3_spacing_um : float
        Micro-channel width and spacing for PPA3.
    channel_extend_left_um, channel_extend_right_um : float
        Extensions for the main channel beyond the strip in X direction.
    channel_split_ratio : float
        Ratio for dividing the main channel width (default 0.5 for even split).
    two_directions, rotate_deg :
        Passed through to fill_with_ppa_grid for all three PPAs.

    Returns
    -------
    (ppa1_reg, ppa2_reg, ppa3_reg, main_channel_reg) : tuple of pya.Region
        ppa1_reg        : PPA polygons inserted for PPA1 block
        ppa2_reg        : PPA polygons inserted for PPA2 block
        ppa3_reg        : PPA polygons inserted for PPA3 block
        main_channel_reg: Region of the single large straight channel polygon
    """
    pya = db
    dbu = layout.dbu

    def um2dbu(x: float) -> int:
        return int(round(x / dbu))

    # Convert sizes to database units
    ppa3_channel_width_um_dbu = um2dbu(ppa3_channel_width_um)
    strip_len_dbu = um2dbu(strip_length_um)
    w1_dbu = um2dbu(rect1_width_um)
    w2_dbu = um2dbu(rect2_width_um)
    chan_dbu = um2dbu(main_channel_width_um)
    chan_side_dbu = um2dbu(side_channel_width_um)
    ch_extend_left_dbu = um2dbu(channel_extend_left_um)
    ch_extend_right_dbu = um2dbu(channel_extend_right_um)
    chan_split_dbu = int(round(chan_dbu * channel_split_ratio))

    ox = um2dbu(origin_x_um)
    oy = um2dbu(origin_y_um)

    orientation = (orientation or "horizontal").lower()

    # --- 1) Define the four boxes: PPA1, PPA2, MAIN CHANNEL, PPA3
    if orientation.startswith("h"):
        # Large channel along X, stack along Y
        x0 = ox
        x1 = ox + strip_len_dbu

        y = oy
        # PPA1 block
        box1 = pya.Box(x0, y, x1, y + w1_dbu)
        y += w1_dbu
        # connecting channel
        ch_box_connect = pya.Box(x0, y, x1, y + ppa3_channel_width_um_dbu)
        y += ppa3_channel_width_um_dbu
        # PPA2 block
        box2 = pya.Box(x0, y, x1, y + w2_dbu)
        y += w2_dbu
        ch_boxmain2 = pya.Box(x0-ch_extend_left_dbu, y, x1, y + chan_split_dbu)
        y += chan_split_dbu
        # Large main channel extension
        ch_boxmain = pya.Box(x0-ch_extend_left_dbu, y, x1+ch_extend_right_dbu, y + chan_split_dbu)
        y += chan_split_dbu

        # PPA3 block
        box3 = pya.Box(x0, y, x1, y + w2_dbu)
        y += w2_dbu

        ch_box_connect2 = pya.Box(x0, y, x1, y + ppa3_channel_width_um_dbu)
        y += ppa3_channel_width_um_dbu
        # PPA2 block
        box4 = pya.Box(x0, y, x1, y + w1_dbu)
        y += w1_dbu
        ch_box2 = pya.Box(x0, y, x1, y + chan_side_dbu)
        y += chan_side_dbu


    # --- 2) Draw the single large channel polygon explicitly
    cell.shapes(layer_index).insert(ch_boxmain)
    main_channel_reg = pya.Region(ch_boxmain)
    cell.shapes(layer_index).insert(ch_boxmain2)
    main_channel_reg2 = pya.Region(ch_boxmain2)
    cell.shapes(layer_index).insert(ch_box2)
    side_channel_reg = pya.Region(ch_box2)
    cell.shapes(layer_index).insert(ch_box_connect)
    connect_channel_reg = pya.Region(ch_box_connect)
    cell.shapes(layer_index).insert(ch_box_connect2)
    connect_channel_reg = pya.Region(ch_box_connect2)
    # --- 3) Fill each PPA box using your existing PPA function

    # PPA1
    ppa1_reg = fill_with_ppa_grid(layout, cell, box1, layer_index,
                                  channel_width_um=ppa1_channel_width_um,
                                  spacing_um=ppa1_spacing_um,
                                  two_directions=two_directions,
                                  rotate_deg=rotate_deg)

    # PPA2
    ppa2_reg = fill_with_variable_channels(layout, cell, box2, layer_index,
                                    min_width_um = 10,
                                    max_width_um = 50,
                                    gap_um= 65,
                                    orientation = "vertical")
    ppa3_reg = fill_with_variable_channels(layout, cell, box3, layer_index,
                                    min_width_um = 10,
                                    max_width_um = 50,
                                    gap_um= 65,
                                    orientation = "vertical")
    # PPA3
    ppa4_reg = fill_with_ppa_grid(layout, cell, box4, layer_index,
                                  channel_width_um=ppa3_channel_width_um,
                                  spacing_um=ppa3_spacing_um,
                                  two_directions=two_directions,
                                  rotate_deg=rotate_deg)

    return ppa1_reg, ppa2_reg, ppa3_reg, main_channel_reg,connect_channel_reg
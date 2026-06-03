"PYTHON SCRIPT FOR MAKING ALTERNATIVE MICROFLUIDIC DESIGNS AND GDS FILES FOR MAKING MASKS"
from microfludesign.format import gds_to_merged_dxf
from microfludesign.geometry import best_spread_points_in_circle, columnar_points_in_circle
import numpy as np
# first version of a mask design

try:
    # Standalone Python (pip install klayout)
    import klayout.db as db
    import klayout.lay as lay  # GUI side (Qt) – only needed if you want a window
    pya = db                   # optional: keep "pya" name compatible with macros
except Exception:
    # Inside KLayout macro runner
    import pya                 # provided by KLayout
    db = pya
from dataclass import AirGapParams, TAirGapParams
from microfludesign.helper import *
# GLOBAL VARIABLES
xcen = 0    # GLOBAL VARIABLE: X-COORDINATE CENTER OF CHIP
ycen = 0    # GLOBAL VARIABLE: Y-COORDINATE CENTER OF CHIP
PATH = r'/Users/bisot/Documents/PostDoc2/microflu_design/microfludesign/designs'
MARGIN = 5 # added margin to deal with overlapping layers

ly = pya.Layout()


top    = ly.create_cell("top")
ground = ly.layer(1,0)
L_PPA = ly.layer(2,0)  # 4 µm height
L_30 = ly.layer(2,0)  # 30 µm height
L_MID = ly.layer(2,0)  # 100 µm height

layers = [L_PPA,L_30,L_MID]
#xcen = 0
#ycen = 0
#SHAPE = top.shapes(ground).insert(Circle(0,0,2*50800,100)) # 2 inch = 50800um, radius of wafer
#MARGIN = 5 # added margin to deal with overlapping layers




#############
### WAFER ###
#############

x = 0
y = 0
SHAPE = top.shapes(ground).insert(Circle(0,0,2*25400,100,x,y)) # 2 inch = 50800um, radius of wafer



ppa_square_size_um = 5000     # side length of the square in the middle of the chip
channel_width_um   = 4.0      # channel width
spacing_um         = 50.0     # gap between channels
rotate_deg         = 0        # rotate the grid if you like (e.g., 45)

dbu = ly.dbu
def um2dbu(x): return int(round(x / dbu))

# Build a square centered at (xcen, ycen) in dbu
half = um2dbu(ppa_square_size_um / 2.0)
cx   = um2dbu(xcen)
cy   = um2dbu(ycen)
ppa_box = pya.Box(cx - half, cy - half, cx + half, cy + half)

# add_facility_markers(ly, top, L_PPA)

variants = [
    6000,
]*12
widths = [1400]*12
positions: List[Tuple[int, int]] = columnar_points_in_circle(len(variants),25400)
positions = [
    (-13000, -15000), (0, -15000), (13000, -15000),
    (-13000,  -6000), (0,  -6000), (13000,  -6000),
    (-13000,   3000), (0,   3000), (13000,   3000),
    (-13000,  12000), (0,  12000), (13000,  12000),
]
for i, (variant,width, (x0, y0)) in enumerate(zip(variants,widths, positions)):
    # give each chip a unique label while keeping other fields the same
    ppa1_reg, ppa2_reg, ppa3_reg, main_ch_reg,_ = final_design(
        layout=ly,
        cell=top,
        layer_index=L_PPA,  # e.g. layout.layer(1, 0)
        strip_length_um=variant,  # along the large channel
        rect1_width_um=width,  # PPA1 thickness
        rect2_width_um=355.0,  # PPA2 thickness
        # rect3_width_um=400.0,  # PPA3 thickness
        main_channel_width_um=250.0,  # single big channel between PPA2 and PPA3
        side_channel_width_um=50.0,  # single big channel between PPA2 and PPA3
        origin_x_um=x0-variant/2,
        origin_y_um=y0,
        orientation="horizontal",
        ppa1_channel_width_um=8.0,
        ppa1_spacing_um=65.0,
        ppa2_channel_width_um=4.0,
        ppa2_spacing_um=50.0,
        ppa3_channel_width_um=8.0,
        ppa3_spacing_um=65.0,
        channel_extend_left_um=1000.0,
        channel_extend_right_um=1500.0,
        channel_split_ratio=0.5,
        two_directions=True,
        rotate_deg=0,
    )
    # break

MergeLayers(top,ly,layers)


ly.write(str(PATH)+"/PPA_simple.gds")
# ly.write(str(PATH)+"/PPA.dxf")

layer_ids = [L_PPA, L_30, L_MID]

# 1) One DXF with all layers (each on its own DXF layer)
in_path = str(PATH)+"/PPA_simple.gds"
out_path = str(PATH)+"/PPA_simple.dxf"
gds_to_merged_dxf(in_path,out_path)
# export_layer_as_hatches_to_dxf(ly, top, L_PPA, out_path, dxf_layer_name="PPA", dxf_color=3)

# Suppose you already have merged_by_spec dict from gds_to_merged_dxf()
"""Render one GLB from the front, turned 25 degrees, with soft studio light (Cycles).

The README's picture of a generated model was made with it. Blender 5.x:

    blender -b -P bench/render_glb.py -- model.glb out.png [width height]
"""

import math
import sys

import bpy  # pyright: ignore[reportMissingImports] - ships with Blender
from mathutils import Vector  # pyright: ignore[reportMissingImports]

argv = sys.argv[sys.argv.index("--") + 1:]
glb, out = argv[0], argv[1]
width, height = (int(argv[2]), int(argv[3])) if len(argv) >= 4 else (1024, 1536)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]

corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
low = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
high = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
center = (low + high) / 2
size = high - low
height_m = size.z

scene = bpy.context.scene
camera_data = bpy.data.cameras.new("camera")
camera_data.lens = 70
camera = bpy.data.objects.new("camera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
# the model faces +Y in Blender (a first render from -Y showed its back); look from the front,
# turned 25 degrees
angle = math.radians(25)
distance = height_m * 2.7
camera.location = center + Vector((math.sin(angle) * distance, math.cos(angle) * distance,
                                   height_m * 0.10))
direction = center - camera.location
camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def area_light(name, location, energy, size_m):
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = energy
    data.size = size_m
    light = bpy.data.objects.new(name, data)
    light.location = location
    light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(light)


unit = max(size.x, size.y, size.z)
area_light("key", center + Vector((1.6, 2.2, 1.8)) * unit, 700 * unit**2, 1.5 * unit)
area_light("fill", center + Vector((-2.0, 1.5, 0.6)) * unit, 250 * unit**2, 2.0 * unit)
area_light("rim", center + Vector((-0.5, -2.2, 1.6)) * unit, 500 * unit**2, 1.0 * unit)

world = bpy.data.worlds.new("world")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.8, 0.8, 0.8, 1.0)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.35
scene.world = world

scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
preferences = bpy.context.preferences.addons["cycles"].preferences
preferences.compute_device_type = "METAL"  # Apple Silicon; CUDA or OPTIX elsewhere
preferences.get_devices()
for device in preferences.devices:
    device.use = True
scene.cycles.samples = 128
scene.cycles.use_denoising = True
scene.render.film_transparent = True
scene.render.resolution_x, scene.render.resolution_y = width, height
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "Standard"
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print(f"rendered {out}: {len(meshes)} meshes, height {height_m:.3f}")

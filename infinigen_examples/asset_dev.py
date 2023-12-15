import argparse
import os
import sys
from pathlib import Path
import itertools
import logging
from copy import copy

logging.basicConfig(
    format='[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] | %(message)s',
    datefmt='%H:%M:%S',
    level=logging.WARNING
)

import bpy
import mathutils

import gin
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pprint import pformat
import imageio

from infinigen.assets.lighting import sky_lighting

from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.camera import spawn_camera, set_active_camera
from infinigen.core.nodes import Nodes, NodeWrangler
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed

from infinigen.core import execute_tasks, surface, init

import bmesh
from infinigen.assets.creatures.util.geometry.nurbs import blender_mesh_from_pydata
import random

from infinigen.terrain import Terrain
from infinigen.assets.weather import kole_clouds
import infinigen.assets.materials.stone as stone
import infinigen.assets.weather.particles as particles2
from infinigen.core.placement import particles
from mathutils import Vector
from infinigen.core.placement import factory

logging.basicConfig(level=logging.INFO)

def extrude(point):
    bpy.ops.mesh.extrude_region_move(MESH_OT_extrude_region={"use_normal_flip":False, "use_dissolve_ortho_edges":False, "mirror":False}, TRANSFORM_OT_translate={"value":point, "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})

def translate(point):
        bpy.ops.transform.translate(value=point, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)

def create_branch(start = (0, 0, 3), level = 2, density = 1, floor=0, strength = 1, color = (1,1,1,1), lateral = 0.15):
    branches = [start]

    iter = len(branches)
    for i in range(iter):
        start = branches.pop(0)
        curr = start
        bpy.ops.mesh.primitive_vert_add()
        translate(start)
        z = start[2]

        if floor != 0:
            floor = random.uniform(floor, z)

        while z > floor:
            delta_x = random.uniform(-lateral, lateral)
            delta_y = random.uniform(-lateral, lateral)
            delta_z = random.uniform(0, 0.05)
            curr = tuple(x + y for x, y in zip(curr, (delta_x, delta_y, -delta_z))) 
            extrude((delta_x, delta_y, -delta_z))
            z -= delta_z
            if random.uniform(0,20) < density:
                branches.append(curr)

    
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.convert(target='CURVE')
    bpy.context.object.data.bevel_depth = 0.005
    

    obj = bpy.context.active_object
    mat = bpy.data.materials.get("Material " + str(level))
    if mat is None:
        mat = bpy.data.materials.new(name="Material " + str(level))
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    emission_node = nodes.new(type='ShaderNodeEmission')
    emission_node.inputs["Strength"].default_value = strength 
    emission_node.inputs["Color"].default_value = color # white color
    links.new(emission_node.outputs[0], nodes['Material Output'].inputs[0])
    obj.data.materials.append(mat)

    return obj, branches

def my_shader(nw: NodeWrangler, params: dict):

    ## TODO: Implement a more complex procedural shader

    noise_texture = nw.new_node(
        Nodes.NoiseTexture, 
        input_kwargs={
            'Scale': params['noise_scale'], 
            'Distortion': params['noise_distortion']
        }
    )
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={'Base Color': noise_texture.outputs["Color"]})
    
    normal = nw.new_node('ShaderNodeNormal')
    
    displacement = nw.new_node('ShaderNodeDisplacement',
        input_kwargs={'Height': noise_texture.outputs["Fac"], 'Scale': 0.02, 'Normal': normal.outputs["Normal"]})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf, 'Displacement': displacement},
        attrs={'is_active_output': True})

class MyAsset(AssetFactory):

    def __init__(self, factory_seed: int, overrides=None):

        super().__init__(factory_seed)
        
        with FixedSeed(factory_seed):
            self.params = self.sample_params()
            if overrides is not None:
                self.params.update(overrides)

    def sample_params(self):
        return {
            'level': 2,
            'decay': 0.25,
            'temperature': 1,
            'lateral': 0.1,
            # TODO: Add more randomized parameters
        }

    def create_asset(self, **_):
        print(self.params)

        level = self.params['level']
        decay = self.params['decay']
        density = 1
        strength = 15
        temperature = self.params['temperature']
        color = (1,1,1,1)

        obj, branches = create_branch(start = (0,0,5), level = level, density = density, floor=0, strength = strength, color = color, lateral = self.params['lateral'])
        
        while len(branches) > 0 and level > 1:
            print("loop")
            temp_branches = []
            iter = len(branches)
            level -= 1
            density *= (1-decay)
            strength = strength**(decay)

            color = (max(color[0] - temperature * 0.3, 0.1), max(color[1] - temperature * 0.3, 0.1), color[2], color[3])
            for i in range(iter):
                curr = branches.pop(0)
                obj, temp = create_branch(start = curr, level = level, density = density, floor = 0.5, strength = strength, color = color, lateral = self.params['lateral'])
                temp_branches.extend(temp)
            branches = temp_branches

        return obj





@gin.configurable
def compose_scene(output_folder, scene_seed, overrides=None, **params):

    ## TODO: Customize this function to arrange your scene, or add other assets
    bpy.ops.preferences.addon_enable(module='ant_landscape')
    bpy.ops.mesh.landscape_add(refresh=True, mesh_size_x=65, mesh_size_y=65)
    terrain = bpy.context.active_object
    stone.apply(terrain)
    
    
    emitter_off = Vector((0, 0, 15))
    rain = particles.particle_system(
            emitter=butil.spawn_plane(location=emitter_off, size=30),
            subject = factory.make_asset_collection(particles2.RaindropFactory(scene_seed), 5),
            settings=particles.rain_settings())
    

    #sky_lighting.add_lighting()

    cam = spawn_camera()
    cam.location = (30,30,0.5)
    cam.rotation_euler = np.deg2rad((90, 0, 135))
    set_active_camera(cam)

    
    factory2 = MyAsset(factory_seed=np.random.randint(0, 1e7))
    if overrides is not None:
        factory2.params.update(overrides)

    factory2.spawn_asset(i=np.random.randint(0, 1e7))
    

    kole_clouds.add_kole_clouds()
    
def iter_overrides(ranges):
    mid_vals = {k: v[len(v)//2] for k, v in ranges.items()}
    for k, v in ranges.items():
        for vi in v:
            res = copy(mid_vals)
            res[k] = vi
            yield res

def create_param_demo(args, seed):

    override_ranges = {
        'level': np.linspace(1, 3, num=3),
        'decay': np.linspace(0, 0.5, num=3),
        'lateral': np.linspace(0.1, 0.3, num=3),
        'temperature': np.linspace(0.1, 1, num=3),
    }
    for i, overrides in enumerate(iter_overrides(override_ranges)):
        
        
        butil.clear_scene()
        print(f'{i=} {overrides=}')
        with FixedSeed(seed):
            compose_scene(args.output_folder, seed, overrides=overrides)
        
        if args.save_blend:
            butil.save_blend(args.output_folder/f'scene_{i}.blend', verbose=True)

        bpy.context.scene.frame_set(i)
        bpy.context.scene.frame_start = i
        bpy.context.scene.frame_end = i
        bpy.ops.render.render(animation=True)

        imgpath = args.output_folder/f'{i:04d}.png'
        img = Image.open(imgpath)
        ImageDraw.Draw(img).text(
            xy=(10, 10), 
            text='\n'.join(f'{k}: {v:.2f}' for k, v in overrides.items()), 
            fill=(76, 252, 85),
            font=ImageFont.load_default(size=50)
        )
        img.save(imgpath)
        

def create_video(args, seed):
    butil.clear_scene()
    with FixedSeed(seed):
        compose_scene(args.output_folder, seed)

    butil.save_blend(args.output_folder/'scene.blend', verbose=True)

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = args.duration_frames
    bpy.ops.render.render(animation=True)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=Path)
    parser.add_argument('--mode', type=str, choices=['param_demo', 'video'])
    parser.add_argument('--duration_frames', type=int, default=1)
    parser.add_argument('--save_blend', action='store_true')
    parser.add_argument('-s', '--seed', default=None, help="The seed used to generate the scene")
    parser.add_argument('-g', '--configs', nargs='+', default=['base'],
                        help='Set of config files for gin (separated by spaces) '
                             'e.g. --gin_config file1 file2 (exclude .gin from path)')
    parser.add_argument('-p', '--overrides', nargs='+', default=[],
                        help='Parameter settings that override config defaults '
                             'e.g. --gin_param module_1.a=2 module_2.b=3')
    parser.add_argument('-d', '--debug', action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)

    args = init.parse_args_blender(parser)
    logging.getLogger("infinigen").setLevel(args.loglevel)

    seed = init.apply_scene_seed(args.seed)
    init.apply_gin_configs(
        configs=args.configs, 
        overrides=args.overrides,
        configs_folder='infinigen_examples/configs'
    )

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 50

    args.output_folder.mkdir(exist_ok=True, parents=True)
    bpy.context.scene.render.filepath = str(args.output_folder.absolute()) + '/'


    if args.mode == 'param_demo':
        create_param_demo(args, seed)
    elif args.mode == 'video':
        create_video(args, seed)
    else:
        raise ValueError(f'Unrecognized {args.mode=}')
    
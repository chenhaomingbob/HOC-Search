import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
from pathlib import Path
from typing import Union
import logging
import numpy as np
import open3d as o3d
from preprocess_ScanNetpp import parse_cls_label, parse_py3d_transform
from utils import transform_ScanNet_to_py3D, alignPclMesh, Rz, shapenet_category_dict
from HOC_search.utils_CAD_retrieval import get_bdb_from_corners, get_corners_of_bb3d_no_index
from HOC_search.ScanNetAnnotation import ScanNetAnnotation, ObjectAnnotation
import pickle
import torch
from pytorch3d.structures import Meshes
import pytorch3d
import copy

from pytorch3d.renderer import (
    RasterizationSettings, MeshRasterizer
)
from HOC_search.Torch3DRenderer.SimpleShader import SimpleShader
from HOC_search.Torch3DRenderer.pytorch3d_rasterizer_custom import MeshRendererViewSelection
from HOC_search.utils_CAD_retrieval import get_bdb_from_corners, get_corners_of_bb3d_no_index, drawOpen3dCylLines
import pytorch3d
from pytorch3d.transforms import Transform3d
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures import Meshes

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

logging.basicConfig(level="INFO")
logger = logging.getLogger('Preprocess ArkitScene')

VALID_CLASS_IDS_200 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
    26,
    27,
    28,
    29,
    31,
    32,
    33,
    34,
    35,
    36,
    38,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    54,
    55,
    56,
    57,
    58,
    59,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    82,
    84,
    86,
    87,
    88,
    89,
    90,
    93,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    110,
    112,
    115,
    116,
    118,
    120,
    121,
    122,
    125,
    128,
    130,
    131,
    132,
    134,
    136,
    138,
    139,
    140,
    141,
    145,
    148,
    154,
    155,
    156,
    157,
    159,
    161,
    163,
    165,
    166,
    168,
    169,
    170,
    177,
    180,
    185,
    188,
    191,
    193,
    195,
    202,
    208,
    213,
    214,
    221,
    229,
    230,
    232,
    233,
    242,
    250,
    261,
    264,
    276,
    283,
    286,
    300,
    304,
    312,
    323,
    325,
    331,
    342,
    356,
    370,
    392,
    395,
    399,
    408,
    417,
    488,
    540,
    562,
    570,
    572,
    581,
    609,
    748,
    776,
    1156,
    1163,
    1164,
    1165,
    1166,
    1167,
    1168,
    1169,
    1170,
    1171,
    1172,
    1173,
    1174,
    1175,
    1176,
    1178,
    1179,
    1180,
    1181,
    1182,
    1183,
    1184,
    1185,
    1186,
    1187,
    1188,
    1189,
    1190,
    1191,
)

CLASS_LABELS_200 = (
    "wall",
    "chair",
    "floor",
    "table",
    "door",
    "couch",
    "cabinet",
    "shelf",
    "desk",
    "office chair",
    "bed",
    "pillow",
    "sink",
    "picture",
    "window",
    "toilet",
    "bookshelf",
    "monitor",
    "curtain",
    "book",
    "armchair",
    "coffee table",
    "box",
    "refrigerator",
    "lamp",
    "kitchen cabinet",
    "towel",
    "clothes",
    "tv",
    "nightstand",
    "counter",
    "dresser",
    "stool",
    "cushion",
    "plant",
    "ceiling",
    "bathtub",
    "end table",
    "dining table",
    "keyboard",
    "bag",
    "backpack",
    "toilet paper",
    "printer",
    "tv stand",
    "whiteboard",
    "blanket",
    "shower curtain",
    "trash can",
    "closet",
    "stairs",
    "microwave",
    "stove",
    "shoe",
    "computer tower",
    "bottle",
    "bin",
    "ottoman",
    "bench",
    "board",
    "washing machine",
    "mirror",
    "copier",
    "basket",
    "sofa chair",
    "file cabinet",
    "fan",
    "laptop",
    "shower",
    "paper",
    "person",
    "paper towel dispenser",
    "oven",
    "blinds",
    "rack",
    "plate",
    "blackboard",
    "piano",
    "suitcase",
    "rail",
    "radiator",
    "recycling bin",
    "container",
    "wardrobe",
    "soap dispenser",
    "telephone",
    "bucket",
    "clock",
    "stand",
    "light",
    "laundry basket",
    "pipe",
    "clothes dryer",
    "guitar",
    "toilet paper holder",
    "seat",
    "speaker",
    "column",
    "bicycle",
    "ladder",
    "bathroom stall",
    "shower wall",
    "cup",
    "jacket",
    "storage bin",
    "coffee maker",
    "dishwasher",
    "paper towel roll",
    "machine",
    "mat",
    "windowsill",
    "bar",
    "toaster",
    "bulletin board",
    "ironing board",
    "fireplace",
    "soap dish",
    "kitchen counter",
    "doorframe",
    "toilet paper dispenser",
    "mini fridge",
    "fire extinguisher",
    "ball",
    "hat",
    "shower curtain rod",
    "water cooler",
    "paper cutter",
    "tray",
    "shower door",
    "pillar",
    "ledge",
    "toaster oven",
    "mouse",
    "toilet seat cover dispenser",
    "furniture",
    "cart",
    "storage container",
    "scale",
    "tissue box",
    "light switch",
    "crate",
    "power outlet",
    "decoration",
    "sign",
    "projector",
    "closet door",
    "vacuum cleaner",
    "candle",
    "plunger",
    "stuffed animal",
    "headphones",
    "dish rack",
    "broom",
    "guitar case",
    "range hood",
    "dustpan",
    "hair dryer",
    "water bottle",
    "handicap bar",
    "purse",
    "vent",
    "shower floor",
    "water pitcher",
    "mailbox",
    "bowl",
    "paper bag",
    "alarm clock",
    "music stand",
    "projector screen",
    "divider",
    "laundry detergent",
    "bathroom counter",
    "object",
    "bathroom vanity",
    "closet wall",
    "laundry hamper",
    "bathroom stall door",
    "ceiling light",
    "trash bin",
    "dumbbell",
    "stair rail",
    "tube",
    "bathroom cabinet",
    "cd case",
    "closet rod",
    "coffee kettle",
    "structure",
    "shower head",
    "keyboard piano",
    "case of water bottles",
    "coat rack",
    "storage organizer",
    "folded chair",
    "fire alarm",
    "power strip",
    "calendar",
    "poster",
    "potted plant",
    "luggage",
    "mattress",
)


def view_selection_new_pose_arkit(scene_name, tmesh, frame_id_pose_dict, frame_id_intrinsic_dict, dist_params,
                                  img_scale, max_views,
                                  silhouette_thres, inst_label_list):
    n_views = 1
    height = 480.  # 参考color
    width = 640.

    view_selection_dict = {}
    img_list = []
    img_path_list = []
    depth_path_list = []
    R_list = []
    T_list = []
    frame_name_list = []
    path_list = []

    raster_settings = RasterizationSettings(
        image_size=(int(height * img_scale), int(width * img_scale)),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        perspective_correct=True,
        clip_barycentric_coords=False,
        cull_backfaces=False
    )
    from tqdm import tqdm
    for frame_name, pose_dict in tqdm(frame_id_pose_dict.items()):
        # print(frame_name, len(frame_id_pose_dict))
        frame_name_list.append(frame_name)
        intrinsics = frame_id_intrinsic_dict[frame_name]

        R_world_to_cam = pose_dict['R'].cpu().numpy()
        T_world_to_cam = pose_dict['T'].cpu().numpy()

        if R_world_to_cam is None or T_world_to_cam is None:
            img_list.append(np.ones((int(height * img_scale), int(width * img_scale))) * -1)
            depth_path_list.append('')
            img_path_list.append('')
            path_list.append('')
            R_list.append(np.zeros((1, 3, 3), dtype=np.float64))
            T_list.append(np.zeros((1, 3), dtype=np.float64))
            continue

        R_list.append(R_world_to_cam)
        T_list.append(T_world_to_cam)

        R = torch.tensor(R_world_to_cam).to(device)
        T = torch.tensor(T_world_to_cam).to(device)

        px, py = (intrinsics[0, 2] * img_scale), (intrinsics[1, 2] * img_scale)
        principal_point = torch.tensor([px, py])[None].type(torch.FloatTensor).to(device)
        principal_point = principal_point.repeat(n_views, 1)
        fx, fy = ((intrinsics[0, 0] * img_scale)), ((intrinsics[1, 1] * img_scale))
        focal_length = torch.tensor([fx, fy])[None].type(torch.FloatTensor).to(device)
        focal_length = focal_length.repeat(n_views, 1)

        cameras = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            in_ndc=False,
            device=device, T=T, R=R,
            image_size=((int(height * img_scale), int(width * img_scale)),))

        renderer = MeshRendererViewSelection(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SimpleShader(
                device=device,
                cameras=cameras,
            )
        )
        tmesh = tmesh.extend(n_views)
        img, fragments = renderer(meshes_world=tmesh.to(device))
        valid_pix = fragments.pix_to_face.repeat(1, 1, 1, 4)

        img[valid_pix < 0] = -1.
        img_ = img.cpu().detach().numpy()[0, :, :, 0]
        img = np.round(img_ * np.max(inst_label_list))
        # if np.max(img)>0:
        # cv2.imshow('image window', img)
        # # add wait key. window waits until user presses a key
        # cv2.waitKey(0)
        # # and finally destroy/close all open windows
        # cv2.destroyAllWindows()
        #
        img_list.append(img)  # 渲染图

    img_ary = np.asarray(img_list)
    # 遍历实例
    for label in inst_label_list:
        view_params_dict = {}

        mask = np.zeros_like(img_ary)
        mask[img_ary == label] = 1
        label_cnt = np.sum(mask, axis=(1, 2))
        label_max = np.max(label_cnt)
        label_norm = label_cnt / label_max
        best_view_idx = np.where(label_norm > silhouette_thres)
        if len(best_view_idx[0]) < 1:
            best_view_idx = np.where(label_norm > 0.)

        if len(best_view_idx[0]) < 4:
            continue

        if len(best_view_idx[0]) < max_views:
            views = best_view_idx[0]
        else:
            views_select = np.linspace(0, len(best_view_idx[0]) - 1, max_views).astype(int)
            views = best_view_idx[0][views_select].astype(int)
        view_params_dict['views'] = views
        view_params_dict['R'] = np.asarray(R_list)[views, :, :]
        view_params_dict['T'] = np.asarray(T_list)[views, :]
        view_params_dict['frame_ids'] = np.asarray(frame_name_list)[views].tolist()

        view_params_dict['intrinsics'] = intrinsics
        view_params_dict['dist_params'] = dist_params
        view_selection_dict[int(label)] = view_params_dict

    return view_selection_dict


def run(
        scene_dir: Union[str, Path],
        output_folder: Union[str, Path],
        # device: Union[str, torch.device] = 'cuda:0',
        render_resolution=(480, 640),
        flip: bool = False,
):
    if isinstance(scene_dir, str):
        scene_dir = Path(scene_dir)
    scene_name = scene_dir.name
    output_dir = scene_dir / output_folder
    if not output_dir.exists():
        output_dir.mkdir()

    mesh_path = scene_dir / "mesh.ply"

    intrinsic_dir = scene_dir / "intrinsic"
    assert intrinsic_dir.exists() and intrinsic_dir.is_dir()
    intrinsic_keys = set(x.stem for x in intrinsic_dir.glob('*.txt'))

    pose_dir = scene_dir / "pose"  # camera to world
    assert pose_dir.exists() and pose_dir.is_dir()
    pose_keys = set(x.stem for x in pose_dir.glob('*.txt'))

    assert pose_keys == intrinsic_keys
    image_keys = sorted(pose_keys)

    inst_seg_base_dir = scene_dir / 'intermediate' / 'scannet200_mask3d_1'

    instance_segmentation_prediction_file = inst_seg_base_dir / 'predictions.txt'
    if not instance_segmentation_prediction_file.exists():
        logger.error(f'No prediction file found in {scene_dir}')
        return

    assert mesh_path.exists()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    T_mat = transform_ScanNet_to_py3D()

    dist_params = np.array([0.0, 0.0, 0.0, 0.0])  # 扰动参数？

    frame_id_pose_dict = {}
    frame_id_intrinsic_dict = {}
    # 加载每一帧的 pose 和 intrinsic
    for frame_id in image_keys:
        pose_dict = {}
        pose_path = pose_dir / f"{frame_id}.txt"
        intrinsic_path = intrinsic_dir / f"{frame_id}.txt"

        pose = np.loadtxt(pose_path)  # (4,4) camera to world
        intrinsic = np.loadtxt(intrinsic_path)  # (3,3)

        world_to_camera = np.linalg.inv(pose)
        T_nviews = np.dot(world_to_camera, np.linalg.inv(T_mat))
        T2 = np.eye(4)
        T2[:3, :3] = Rz(np.deg2rad(180))
        T_nviews = np.dot(T2, T_nviews)

        rot_py3d = np.copy(T_nviews[:3, :3].T)
        T_nviews[:3, :3] = rot_py3d

        R_world_to_cam = np.expand_dims(T_nviews, axis=0)[:, 0:3, 0:3]
        T_world_to_cam = np.expand_dims(T_nviews, axis=0)[:, 0:3, 3]

        R = torch.tensor(R_world_to_cam)
        T = torch.tensor(T_world_to_cam)
        pose_dict['R'] = R
        pose_dict['T'] = T

        frame_id_pose_dict[frame_id] = pose_dict
        frame_id_intrinsic_dict[frame_id] = intrinsic
    # 加载实例分割结果
    with open(instance_segmentation_prediction_file) as f:
        instances = [x.strip().split(' ') for x in f.readlines()]
    ##################
    points = np.copy(np.asarray(mesh.vertices))
    inst_label_list = []
    inst_label_map = np.zeros((points.shape[0], 1))
    valid_annotations = []
    for i, inst_items in enumerate(instances):
        # check confidence
        if float(inst_items[2]) < 0.5:
            continue

        filepath = inst_seg_base_dir / inst_items[0]
        inst_mask = np.loadtxt(filepath).astype(bool)

        inst_scannet_label_id = int(inst_items[1])
        inst_scannet_label_name = CLASS_LABELS_200[int(VALID_CLASS_IDS_200.index(inst_scannet_label_id))]
        inst_shapenet_label_name = parse_cls_label(inst_scannet_label_name)

        # 根据类别过滤
        if inst_shapenet_label_name is None or inst_shapenet_label_name not in shapenet_category_dict.keys():
            continue
        inst_id = i
        inst_label_map[inst_mask] = inst_id
        inst_label_list.append(i)  # 实例id
        valid_annotations.append({
            'object_id': inst_id,
            'inst_mask': inst_mask,
            'shapenet_cls_label': inst_shapenet_label_name,
            'segments': np.where(inst_mask == True)[0]
        })

    mesh_pytorch3d = alignPclMesh(mesh, T=T_mat)

    mesh_pytorch3d_bg = copy.deepcopy(mesh_pytorch3d)
    indices = np.where(inst_label_map > 0)[0].tolist()
    mesh_pytorch3d_bg.remove_vertices_by_index(indices)

    tmesh = Meshes(
        verts=[torch.tensor(np.asarray(mesh_pytorch3d.vertices)[:, :3].astype(np.float32))],
        faces=[torch.tensor(np.asarray(mesh_pytorch3d.triangles))]
    )
    inst_label_norm = inst_label_map / np.max(np.unique(inst_label_list))
    tex = torch.tensor(inst_label_norm.astype(np.float32)).unsqueeze(dim=0)
    tex = tex.repeat(1, 1, 3)
    tmesh.textures = pytorch3d.renderer.mesh.textures.TexturesVertex(verts_features=tex)

    view_sel_path = output_dir / 'view_selection.pkl'
    logger.info('Start view selection')
    if os.path.exists(view_sel_path):
        pkl_file = open(view_sel_path, 'rb')
        view_selection_dict = pickle.load(pkl_file)
        pkl_file.close()
    else:
        # parameters for view selection
        img_scale = 1.  # align rgb image and depth image
        # img_scale = .4
        max_views = 30
        silhouette_thres = 0.3

        view_selection_dict = view_selection_new_pose_arkit(scene_name, tmesh, frame_id_pose_dict,
                                                            frame_id_intrinsic_dict,
                                                            dist_params, img_scale, max_views, silhouette_thres,
                                                            inst_label_list)
    if view_selection_dict is None:
        return

    out_file = open(view_sel_path, 'wb')
    pickle.dump(view_selection_dict, out_file)
    out_file.close()

    logger.info('View selection done')

    all_boxes_selected = None
    obj_3d_list = []
    mesh_vertices = np.array(mesh_pytorch3d.vertices)
    obj_indices = []
    for obj_count, inst_items in enumerate(valid_annotations):
        inst_mask = inst_items['inst_mask']  #
        # np.where(inst_mask > 0)[0].tolist()

        inst_vertices = mesh_vertices[inst_mask, :]  #

        # 创建对应的obb
        aabb_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(inst_vertices))
        obb_bbox = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(aabb_bbox)
        # obb_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(inst_vertices))

        # obb_bbox.rotate(T_mat[0:3, 0:3], center=(0, 0, 0))  # 旋转到pytorch3d的坐标系

        box_final = get_corners_of_bb3d_no_index(obb_bbox.R.T, obb_bbox.extent / 2, obb_bbox.center)
        center_final, basis_final, coeffs_final = get_bdb_from_corners(box_final)

        line_color = [0, 1, 0]
        lineSets_selected = drawOpen3dCylLines([box_final], line_color)

        if all_boxes_selected is None:
            all_boxes_selected = lineSets_selected
        else:
            all_boxes_selected += lineSets_selected

        transform3d, transform_dict = parse_py3d_transform(center_final, basis_final, coeffs_final * 2)

        object_id = inst_items['object_id']
        shapenet_cls_label = inst_items['shapenet_cls_label']
        if object_id in view_selection_dict:
            view_params = view_selection_dict[object_id]
            catid_cad = shapenet_category_dict[shapenet_cls_label]

            annotation = {
                'segments': inst_items['segments'].tolist()
            }

            obj_instance = ObjectAnnotation(
                object_id,
                shapenet_cls_label,
                scannet_category_label=None,
                view_params=view_params,
                transform3d=transform3d,
                transform_dict=transform_dict,
                catid_cad=catid_cad,
                scan2cad_annotation_dict=annotation
            )
            obj_3d_list.append(obj_instance)

    # 所有bbox
    bpaResult = o3d.io.write_triangle_mesh(output_dir / "bbox_all.ply",
                                           all_boxes_selected)
    # mesh + bbox
    bbox_and_mesh = o3d.io.write_triangle_mesh(output_dir / "mesh_py3d_with_bbox_all.ply",
                                               all_boxes_selected + mesh_pytorch3d)

    bg = o3d.io.write_triangle_mesh(output_dir / "bg.ply", mesh_pytorch3d_bg)

    scene_obj = ScanNetAnnotation(scene_name, obj_3d_list, inst_label_map, scene_type=None)

    pkl_out_path = output_dir / f'{scene_name}.pkl'

    if scene_obj is not None:
        pkl_out_file = open(pkl_out_path, 'wb')
        pickle.dump(scene_obj, pkl_out_file)
        pkl_out_file.close()

    print('Extracting scene information done for ' + str(scene_name))


def arg_parser():
    parser = argparse.ArgumentParser(description='Mask3D Segmentation')
    parser.add_argument(
        '--workspace',
        type=str,
        required=True,
        help=
        'Path to workspace directory. There should be a "mesh.ply" file and "pose" folder inside.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='intermediate/HOC_Search',
        help=
        'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version.'
    )
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--config', help='Name of config file')
    parser.add_argument(
        '--flip',
        action="store_true",
        help='Mirror the input mesh file, this is part of test time augmentation.',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()

    # setup_seeds(seed=args.seed)
    run(scene_dir=args.workspace, output_folder=args.output, flip=args.flip)

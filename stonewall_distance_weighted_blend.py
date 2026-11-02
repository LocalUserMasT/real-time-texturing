import time

from pathlib import Path

import glfw

import moderngl

import numpy as np

import trimesh

from PIL import Image

WIDTH, HEIGHT = 1000, 800

MOVE_SPEED = 0.5

MOUSE_SENSITIVITY = 0.12

FOV_DEGREES = 45.0

def normalize(v):

    n = np.linalg.norm(v)

    return v / n if n > 1e-8 else v

def perspective(fov_y, aspect, near, far):

    f = 1.0 / np.tan(fov_y / 2.0)

    return np.array([

        [f / aspect, 0, 0, 0],

        [0, f, 0, 0],

        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],

        [0, 0, -1, 0],

    ], dtype=np.float32)

def look_at(eye, target, up):

    forward = normalize(target - eye)

    right = normalize(np.cross(forward, up))

    camera_up = np.cross(right, forward)

    matrix = np.eye(4, dtype=np.float32)

    matrix[0, :3] = right

    matrix[1, :3] = camera_up

    matrix[2, :3] = -forward

    matrix[0, 3] = -np.dot(right, eye)

    matrix[1, 3] = -np.dot(camera_up, eye)

    matrix[2, 3] = np.dot(forward, eye)

    return matrix

def load_colmap_cameras(images_txt_path):

    if not images_txt_path.is_file():

        raise FileNotFoundError(images_txt_path)

    lines = [line.strip() for line in images_txt_path.read_text().splitlines()

             if line.strip() and not line.lstrip().startswith('#')]

    if len(lines) % 2:

        raise ValueError('images.txt の画像情報行とPOINTS2D行の組が不正です')

    cameras = []

    for k in range(0, len(lines), 2):

        parts = lines[k].split()

        if len(parts) < 10:

            raise ValueError(f'不正なimages.txt行: {lines[k][:100]}')

        q = np.array(list(map(float, parts[1:5])), dtype=np.float64)

        q /= np.linalg.norm(q)

        w, x, y, z = q

        R = np.array([

            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],

            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],

            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],

        ], dtype=np.float64)

        t = np.array(list(map(float, parts[5:8])), dtype=np.float64)

        cameras.append(dict(name=' '.join(parts[9:]), camera_id=int(parts[8]),

                            R=R, t=t, center=-R.T @ t))

    if len(cameras) < 2:

        raise ValueError('2台以上の撮影Cameraが必要です')

    return cameras

def load_intrinsics(path):

    result = {}

    for line in path.read_text().splitlines():

        if not line.strip() or line.lstrip().startswith('#'):

            continue

        parts = line.split()

        cid, model, width, height = int(parts[0]), parts[1], int(parts[2]), int(parts[3])

        params = list(map(float, parts[4:]))

        if model != 'SIMPLE_RADIAL' or len(params) != 4:

            raise ValueError(f'未対応のCameraモデル: {model} (ID={cid})')

        f, cx, cy, k1 = params

        result[cid] = dict(width=width, height=height, f=f, cx=cx, cy=cy, k1=k1)

    return result

def nearest_two_cameras(user_position_colmap, cameras):

    distances = np.array([np.linalg.norm(user_position_colmap - c['center']) for c in cameras])

    indices = np.argsort(distances, kind='stable')[:2]

    return [(cameras[int(i)], float(distances[i])) for i in indices]

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR.parent

cameras = load_colmap_cameras(DATA_DIR / 'sparse_txt' / 'images.txt')

intrinsics = load_intrinsics(DATA_DIR / 'sparse_txt' / 'cameras.txt')

print(f'COLMAP cameras: {len(cameras)}')

mesh = trimesh.load(BASE_DIR / 'scene_mesh.ply', process=False)

if not isinstance(mesh, trimesh.Trimesh):

    raise TypeError('scene_mesh.ply が三角形Meshではありません')

vertices = np.asarray(mesh.vertices, dtype=np.float32)

faces = np.asarray(mesh.faces, dtype=np.int32)

print('vertices:', vertices.shape, 'faces:', faces.shape)

triangle_vertices = vertices[faces].reshape(-1, 3).astype(np.float32)

mesh_center = triangle_vertices.mean(axis=0)

triangle_vertices -= mesh_center

mesh_scale = float(np.max(np.linalg.norm(triangle_vertices, axis=1)))

if mesh_scale <= 0:

    raise ValueError('Mesh scale is zero')

triangle_vertices /= mesh_scale

if not glfw.init():

    raise RuntimeError('GLFW init failed')

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)

glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

window = glfw.create_window(WIDTH, HEIGHT, 'Stonewall Projective Texturing - Distance Weighted Blend', None, None)

if not window:

    glfw.terminate()

    raise RuntimeError('GLFW window creation failed')

glfw.make_context_current(window)

glfw.swap_interval(1)

try:

    ctx = moderngl.create_context()

    print('OpenGL:', ctx.info['GL_VERSION'])

    print('Renderer:', ctx.info['GL_RENDERER'])

    # RGB描画用: 撮影Cameraからの可視性をDepth Mapと比較
    program = ctx.program(
        vertex_shader="""
            #version 410
            in vec3 in_position;
            uniform mat4 mvp;
            uniform vec3 mesh_center;
            uniform float mesh_scale;
            out vec3 world_position;
            void main() {
                world_position = in_position * mesh_scale + mesh_center;
                gl_Position = mvp * vec4(in_position, 1.0);
            }
        """,
        fragment_shader="""
            #version 410
            in vec3 world_position;
            out vec4 fragColor;
            uniform mat3 source_R1, source_R2;
            uniform vec3 source_t1, source_t2;
            uniform vec4 intrinsics1, intrinsics2;
            uniform vec2 image_size1, image_size2;
            uniform sampler2D source_image1, source_image2;
            uniform sampler2D source_depth1, source_depth2;
            uniform float occlusion_bias;
            uniform vec2 source_distances;  // 仮想カメラから選択された2台までの距離

            bool project_source(vec3 wp, mat3 R, vec3 t, vec4 intr,
                                vec2 size, out vec2 uv, out float depth_z) {
                vec3 pc = R * wp + t;
                depth_z = pc.z;
                if (pc.z <= 0.0) return false;
                vec2 xy = pc.xy / pc.z;
                float r2 = dot(xy, xy);
                vec2 px = intr.x * xy * (1.0 + intr.w * r2) + intr.yz;
                if (any(isnan(px)) || any(isinf(px)) ||
                    px.x < 0.0 || px.y < 0.0 ||
                    px.x >= size.x || px.y >= size.y) return false;
                uv = px / size;
                return true;
            }

            bool visible_from(vec3 wp, mat3 R, vec3 t, vec4 intr,
                              vec2 size, sampler2D depth_map, out vec2 uv) {
                float z;
                if (!project_source(wp, R, t, intr, size, uv, z))
                    return false;
                ivec2 pixel = clamp(ivec2(uv * size), ivec2(0), ivec2(size) - 1);
                float surface_z = texelFetch(depth_map, pixel, 0).r;
                // 空の画素は無効。手前の面と深さがほぼ一致する場合のみ可視
                if (surface_z <= 0.0 || surface_z > 1.0e8) return false;
                float bias = max(occlusion_bias, 0.002 * z);
                return z <= surface_z + bias;
                
            }

            void main() {
                vec2 uv1, uv2;
                bool visible1 = visible_from(world_position, source_R1, source_t1,
                                             intrinsics1, image_size1, source_depth1, uv1);
                bool visible2 = visible_from(world_position, source_R2, source_t2,
                                             intrinsics2, image_size2, source_depth2, uv2);

                if (visible1 && visible2) {
                    // 距離の逆数を正規化した重み。距離0でも安全に動作する。
                    float eps = 1e-5;
                    float inv1 = 1.0 / max(source_distances.x, eps);
                    float inv2 = 1.0 / max(source_distances.y, eps);
                    float w1 = inv1 / (inv1 + inv2);
                    float w2 = 1.0 - w1;
                    vec3 color1 = texture(source_image1, uv1).rgb;
                    vec3 color2 = texture(source_image2, uv2).rgb;
                    fragColor = vec4(w1 * color1 + w2 * color2, 1.0);
                } else if (visible1) {
                    fragColor = vec4(texture(source_image1, uv1).rgb, 1.0);
                } else if (visible2) {
                    fragColor = vec4(texture(source_image2, uv2).rgb, 1.0);
                } else {
                    // どちらの画像からも見えない領域は赤色で表示
                    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
                }
            }
        """,
    )

    # 撮影Camera視点の深度画像を作るShader
    # source画像と同じSIMPLE_RADIAL射影を頂点に適用する
    depth_program = ctx.program(
        vertex_shader="""
            #version 410
            in vec3 in_position;
            uniform vec3 mesh_center;
            uniform float mesh_scale;
            uniform mat3 source_R;
            uniform vec3 source_t;
            uniform vec4 intrinsics;
            uniform vec2 image_size;
            uniform float near_plane;
            uniform float far_plane;
            out float camera_z;

            void main() {
                vec3 wp = in_position * mesh_scale + mesh_center;
                vec3 pc = source_R * wp + source_t;
                camera_z = pc.z;
                float z = max(pc.z, 0.000001);
                vec2 xy = pc.xy / z;
                float r2 = dot(xy, xy);
                vec2 px = intrinsics.x * xy * (1.0 + intrinsics.w * r2)
                          + intrinsics.yz;
                vec2 ndc = px / image_size * 2.0 - 1.0;
                float A = (far_plane + near_plane) / (far_plane - near_plane);
                float B = -2.0 * far_plane * near_plane / (far_plane - near_plane);
                gl_Position = vec4(ndc * z, A * z + B, z);
            }
        """,
        fragment_shader="""
            #version 410
            in float camera_z;
            layout(location = 0) out float depth_value;
            void main() { depth_value = camera_z; }
        """,
    )

    program['mesh_center'].value = tuple(float(v) for v in mesh_center)
    program['mesh_scale'].value = mesh_scale
    program['source_image1'].value = 0
    program['source_image2'].value = 1
    program['source_depth1'].value = 2
    program['source_depth2'].value = 3
    program['occlusion_bias'].value = max(mesh_scale * 0.002, 0.0001)
    program['source_distances'].value = (1.0, 1.0)
    depth_program['mesh_center'].value = tuple(float(v) for v in mesh_center)
    depth_program['mesh_scale'].value = mesh_scale
    depth_program['near_plane'].value = max(mesh_scale * 0.0001, 0.001)
    depth_program['far_plane'].value = max(mesh_scale * 100.0, 1000.0)

    vbo = ctx.buffer(triangle_vertices.tobytes())

    vao = ctx.vertex_array(program, [(vbo, '3f', 'in_position')])

    depth_vao = ctx.vertex_array(depth_program, [(vbo, '3f', 'in_position')])
    ctx.enable(moderngl.DEPTH_TEST)
    depth_textures = [None, None]
    depth_fbos = [None, None]


    # 最初の撮影Cameraの位置と向きを初期視点にする
    initial_cam = cameras[0]

    # COLMAP座標系から、表示用の正規化座標系に変換
    initial_pos = (initial_cam['center'] - mesh_center) / mesh_scale

    # 描画時にZ軸周り180°回転しているため、逆変換
    camera_position = np.array(
        [-initial_pos[0], -initial_pos[1], initial_pos[2]],
        dtype=np.float32
    )

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # COLMAPのCameraが向いている方向を取得
    forward_colmap = initial_cam['R'].T @ np.array([0.0, 0.0, 1.0])

    # 表示座標系へ変換
    forward = normalize(np.array([
        -forward_colmap[0],
        -forward_colmap[1],
        forward_colmap[2]
    ], dtype=np.float32))

    yaw = np.degrees(np.arctan2(forward[2], forward[0]))
    pitch = np.degrees(np.arcsin(np.clip(forward[1], -1.0, 1.0)))

    mouse_state = {
        'yaw': float(yaw),
        'pitch': float(pitch),
        'last': None,
        'captured': True
    }

    def cursor_callback(win, xpos, ypos):

        if not mouse_state['captured']:

            return

        if mouse_state['last'] is None:

            mouse_state['last'] = (xpos, ypos)

            return

        dx = xpos - mouse_state['last'][0]

        dy = mouse_state['last'][1] - ypos

        mouse_state['last'] = (xpos, ypos)

        mouse_state['yaw'] += dx * MOUSE_SENSITIVITY

        mouse_state['pitch'] = float(np.clip(mouse_state['pitch'] + dy * MOUSE_SENSITIVITY, -89.0, 89.0))

    glfw.set_cursor_pos_callback(window, cursor_callback)

    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    active_sources = [None, None]

    source_textures = [None, None]

    def activate_source(cam, slot):

        if active_sources[slot] == cam['name']:

            return

        image_path = DATA_DIR / 'images' / cam['name']

        if not image_path.is_file():

            raise FileNotFoundError(f'撮影画像が見つかりません: {image_path}')

        with Image.open(image_path) as img:

            rgb = img.convert('RGB')

            width, height = rgb.size

            raw = rgb.tobytes()

        intr = intrinsics[cam['camera_id']]

        if (width, height) != (intr['width'], intr['height']):

            raise ValueError(f'{cam["name"]}: 画像サイズがcameras.txtと一致しません')

        tex = ctx.texture((width, height), 3, raw)

        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        if source_textures[slot] is not None:

            source_textures[slot].release()

        source_textures[slot] = tex
        # 撮影Cameraの深度マップを生成（撮影画像の切替時のみ）
        if depth_fbos[slot] is not None:
            depth_fbos[slot].release()
            depth_fbos[slot] = None
        if depth_textures[slot] is not None:
            depth_textures[slot].release()
            depth_textures[slot] = None
        depth_tex = ctx.texture((width, height), 1, dtype='f4')
        depth_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        depth_attachment = ctx.depth_renderbuffer((width, height))
        depth_fbo = ctx.framebuffer(
            color_attachments=[depth_tex], depth_attachment=depth_attachment
        )
        depth_textures[slot] = depth_tex
        depth_fbos[slot] = depth_fbo
        depth_program['source_R'].write(cam['R'].T.astype('f4').tobytes())
        depth_program['source_t'].value = tuple(float(v) for v in cam['t'])
        depth_program['intrinsics'].value = (
            intr['f'], intr['cx'], intr['cy'], intr['k1']
        )
        depth_program['image_size'].value = (float(width), float(height))
        depth_fbo.use()
        ctx.viewport = (0, 0, width, height)
        depth_fbo.clear(red=1.0e9, green=0.0, blue=0.0, alpha=1.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        depth_vao.render(moderngl.TRIANGLES)
        ctx.screen.use()

        suffix = str(slot + 1)

        program['source_R' + suffix].write(cam['R'].T.astype('f4').tobytes())

        program['source_t' + suffix].value = tuple(float(v) for v in cam['t'])

        program['intrinsics' + suffix].value = (intr['f'], intr['cx'], intr['cy'], intr['k1'])

        program['image_size' + suffix].value = (float(width), float(height))

        active_sources[slot] = cam['name']

        print(f'RGB source {slot+1}: {cam["name"]}')

    print('¥nマウス: 視線変更 / W,S,A,D: 移動 / Space,Shift: 上下 / Tab: 捕捉切替 / Esc: 終了')

    print('Occlusion test: 深度マップで遮蔽判定し、不可視なら2台目を試します。¥n')

    previous_time = time.perf_counter()

    last_print_time = previous_time

    tab_was_pressed = False

    while not glfw.window_should_close(window):

        now = time.perf_counter()

        dt = min(now - previous_time, 0.1)

        previous_time = now

        glfw.poll_events()

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:

            glfw.set_window_should_close(window, True)

        tab_pressed = glfw.get_key(window, glfw.KEY_TAB) == glfw.PRESS

        if tab_pressed and not tab_was_pressed:

            mouse_state['captured'] = not mouse_state['captured']

            mouse_state['last'] = None

            glfw.set_input_mode(window, glfw.CURSOR,

                                glfw.CURSOR_DISABLED if mouse_state['captured'] else glfw.CURSOR_NORMAL)

        tab_was_pressed = tab_pressed

        yaw_rad = np.radians(mouse_state['yaw'])

        pitch_rad = np.radians(mouse_state['pitch'])

        camera_front = normalize(np.array([

            np.cos(yaw_rad) * np.cos(pitch_rad),

            np.sin(pitch_rad),

            np.sin(yaw_rad) * np.cos(pitch_rad),

        ], dtype=np.float32))

        right = normalize(np.cross(camera_front, world_up))

        velocity = MOVE_SPEED * dt

        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:

            camera_position += camera_front * velocity

        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:

            camera_position -= camera_front * velocity

        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:

            camera_position -= right * velocity

        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:

            camera_position += right * velocity

        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:

            camera_position += world_up * velocity

        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:

            camera_position -= world_up * velocity

        fb_width, fb_height = glfw.get_framebuffer_size(window)

        if fb_width == 0 or fb_height == 0:

            continue

        ctx.viewport = (0, 0, fb_width, fb_height)

        projection = perspective(np.radians(FOV_DEGREES), fb_width / fb_height, 0.01, 100.0)

        view = look_at(camera_position, camera_position + camera_front, world_up)

        # Z軸周りに180°回転

        rotation = np.eye(4, dtype=np.float32)

        rotation[0, 0] = -1.0

        rotation[1, 1] = -1.0

        mvp = projection @ view @ rotation

        program['mvp'].write(mvp.T.astype('f4').tobytes())

        user_position_colmap = (

            rotation[:3, :3].T @ camera_position

        ) * mesh_scale + mesh_center

        selected = nearest_two_cameras(user_position_colmap.astype(np.float64), cameras)
        program['source_distances'].value = (selected[0][1], selected[1][1])

        activate_source(selected[0][0], 0)

        activate_source(selected[1][0], 1)

        source_textures[0].use(location=0)

        source_textures[1].use(location=1)
        depth_textures[0].use(location=2)
        depth_textures[1].use(location=3)
        ctx.screen.use()
        ctx.viewport = (0, 0, fb_width, fb_height)

        ctx.clear(0.08, 0.08, 0.08, 1.0, depth=1.0)

        vao.render(moderngl.TRIANGLES)

        glfw.swap_buffers(window)

        if now - last_print_time >= 0.5:

            print('Nearest 1:', selected[0][0]['name'], '| Nearest 2:', selected[1][0]['name'])

            last_print_time = now

finally:

    glfw.terminate()

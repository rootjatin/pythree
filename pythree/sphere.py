import math

def create_sphere(radius, num_steps):
    vertices = []
    faces = []
    theta_step = math.pi / num_steps
    phi_step = 2 * math.pi / num_steps

    for i in range(num_steps+1):
        theta = i * theta_step
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        for j in range(num_steps+1):
            phi = j * phi_step
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)

            x = radius * sin_theta * cos_phi
            y = radius * sin_theta * sin_phi
            z = radius * cos_theta
            vertices.append([x, y, z])

            if i < num_steps and j < num_steps:
                v1 = i * (num_steps+1) + j
                v2 = v1 + 1
                v3 = (i+1) * (num_steps+1) + j
                v4 = v3 + 1
                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])

    return vertices, faces

def save_obj_file(filename, vertices, faces):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        
        for face in faces:
            f.write('f {} {} {}\n'.format(face[0]+1, face[1]+1, face[2]+1))



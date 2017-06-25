import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pandas as pd
import scipy.sparse as spsp
import scipy.sparse.linalg as spspl
import scipy.linalg as spl
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
import scipy as sc
import os
import sys, argparse


def plot_triangular_solution(mesh, solution, omega, save_fig=False, save_dir="Plots"):
    x, y = mesh[0][:, 0], mesh[0][:, 1]
    triangles = mesh[1]
    triangulation = tri.Triangulation(x, y, triangles)
    refiner = tri.UniformTriRefiner(triangulation)
    tri_refi, z_test_refi = refiner.refine_field(solution, subdiv=0)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.use_sticky_edges = False
    ax.margins(0.07)
    ax.set_xlabel("Omega = {}".format(str(omega)[0:4]))
    plt.tripcolor(triangulation, solution)
    plt.colorbar()
    plt.title("Solution de l'équation de Helmholtz pour {} triangles".format(len(mesh[1])))
    if save_fig:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        list_img = [int(k.split('.png')[0]) for k in os.listdir(save_dir)]
        if len(list_img) == 0:
            k = 1
        else:
            k = max(list_img) + 1
        plt.savefig("{}/{}.png".format(save_dir, k))

def read_mesh_file(indice, filename="maillages.zip", logdir="Maillages/"):
    # Lis un fichier de maillage par l'indice
    import os
    import zipfile
    if not os.path.exists(logdir + filename):
        print("Mesh file (%s) is missing" % (logdir + filename))
    else:
        with zipfile.ZipFile(logdir + filename, 'r') as zipext:
            if "maillage{}.mesh".format(indice) in zipext.namelist():
                data = zipext.read("maillage{}.mesh".format(indice)).decode('UTF-8')
                return data.split()
            else:
                raise ValueError("Could not find maillage{}.mesh in archive".format(indice))


def create_mesh(data):
    # Crée le maillage : un tableau des sommets et un tableau de triangles
    nb_vertices = int(data[3])
    print("-- Number of vertices : %s" % nb_vertices)
    vertices = np.zeros([nb_vertices, 3])
    for k in range(nb_vertices):
        vertices[k] = [float(data[7 + 3 * k]), float(data[7 + 3 * k + 1]), float(data[7 + 3 * k + 2])]

    offset = 7 + 3 * nb_vertices + 3
    nb_triangles = int(data[offset])
    print("-- Number of triangles : %s" % nb_triangles)
    triangles = np.zeros([nb_triangles, 3], dtype=np.int32)
    offset = 7 + 3 * nb_vertices + 10
    for k in range(nb_triangles):
        triangles[k] = [int(data[offset + 3 * k]), int(data[offset + 3 * k + 1]), int(data[offset + 3 * k + 2])]
    return vertices, triangles-1


def get_outer_points(mesh):
    # Renvoie les points sur la frontière
    voisins = {}
    for som1, som2, som3 in mesh[1]:
        tab = [som1, som2, som3]
        for k in range(3):
            key1 = str(tab[k]) + '-' + str(tab[(k + 1) % 3])
            key2 = str(tab[(k + 1) % 3]) + '-' + str(tab[k])
            if key1 in voisins.keys():
                voisins[key1] += 1
            else:
                if key2 in voisins.keys():
                    voisins[key2] += 1
                else:
                    voisins[key1] = 1
    sommets_out = []
    arrete_out = list()
    for key in voisins.keys():
        if voisins[key] == 1:
            a, b = key.split("-")
            arrete_out.append([int(a), int(b)])
            if int(a) not in sommets_out:
                sommets_out.append(int(a))
            if int(b) not in sommets_out:
                sommets_out.append(int(b))
    return np.array(sommets_out), np.array(arrete_out)

def plot_mesh(mesh):
    # Affiche le maillage passé en argument
    x, y = mesh[0][:, 0], mesh[0][:, 1]
    triangles = mesh[1]
    triangulation = tri.Triangulation(x, y, triangles)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.use_sticky_edges = False
    ax.margins(0.07)
    ax.triplot(triangulation, alpha=1, linewidth=1, color='k')
    ax.set_title("Affichage du maillage pour {} triangles".format(len(mesh[1])))
    plt.show()

def aire(sommets_tab):
    s1,s2,s3=sommets_tab
    x1,y1,z1 = s1
    x2,y2,z2 = s2
    x3,y3,z3 = s3
    return 0.5*abs((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1))

def k_elem(q,l,m,mesh):
    s123 = [mesh[0][k] for k in mesh[1][q]]
    return np.dot((s123[(l+2)%3]-s123[(l+1)%3]),(s123[(m+2)%3]-s123[(m+1)%3]))/(4*aire(s123))

def m_elem(q,l,m,mesh):
    return (aire([mesh[0][k] for k in mesh[1][q]])/12)*(1+int(l==m))

def a_elem(q,l,m,mesh, omega):
    return k_elem(q,l,m,mesh)-omega*omega*m_elem(q,l,m,mesh)

def calc_K(mesh):
    K = np.zeros([len(mesh[0]), len(mesh[0])])
    for q in range(len(mesh[1])):
        for l in range(3):
            for m in range(3):
                K[mesh[1][q][l], mesh[1][q][m]] += k_elem(q, l, m, mesh)
    return K

def calc_M(mesh):
    M = np.zeros([len(mesh[0]), len(mesh[0])])
    for q in range(len(mesh[1])):
        for l in range(3):
            for m in range(3):
                M[mesh[1][q][l], mesh[1][q][m]] += m_elem(q, l, m, mesh)
    return M


def calc_A(mesh, omega):
    triangles = mesh[1]
    i, j, val = list(), list(), list()
    outers_som, outers_arrete = get_outer_points(mesh)
    for q in range(len(mesh[1])):
        for l in range(3):
            for m in range(3):
                i.append(triangles[q][l])
                j.append(triangles[q][m])
                val.append(a_elem(q,l,m,mesh, omega))
    return csr_matrix((val, (i, j)), dtype=np.complex_,shape=(len(mesh[0]),len(mesh[0])))


def calc_F(mesh, omega):
    # Calcul le second membre de la méthode de galerkin
    vertices, triangles = mesh
    Ns, Nt = len(mesh[0]), len(mesh[1])
    d = np.array([np.cos(np.pi*alpha/180), np.sin(np.pi*alpha/180), 0])
    clim = lambda x,n: np.dot(d, n) * np.exp(1j * omega * np.dot(d, x))
    outer_sommets, outer_arretes = get_outer_points(mesh)
    F = np.zeros(len(mesh[0]))
    for point in outer_sommets:
        arretes = [outer_arretes[k] for k, elem in enumerate([point in tab for tab in outer_arretes]) if elem]
        for arrete in arretes:
            s0,s1 = vertices[arrete[0]],vertices[arrete[1]]
            normale = np.cross(s1-s0,np.array([0,0,1]))
            normale /= np.linalg.norm(normale)
            F[point]+= (np.linalg.norm(s1-s0)/2)*clim(0.5*(s0+s1),normale)
    return F


def valeur_propre(mesh):
    K, M = calc_K(mesh), calc_M(mesh)
    print(K.shape)
    u, v = spspl.eigsh(K, M=M, which='SM')
    return u, v


def solve_resonance(mesh, save_fig=False):
    # Résouds le problème aux valeurs propres généralisées
    save_dir = "Plots"
    eigenvalues, eigenvectors = valeur_propre(mesh)
    fig, axarray = plt.subplots(2, 3)
    x, y = mesh[0][:, 0], mesh[0][:, 1]
    triangles = mesh[1]
    triangulation = tri.Triangulation(x, y, triangles)
    j = 0
    for i in range(2):
        for k in range(3):
            if i == 0:
                j = k + i
            else:
                j = k + i + 2
            axarray[i, k].set_aspect('equal')
            axarray[i, k].use_sticky_edges = False
            axarray[i, k].margins(0.07)
            axarray[i, k].set_title("VP = {}".format(eigenvalues[j]))
            axarray[i, k].tripcolor(triangulation, eigenvectors[:, j])
            fig.colorbar(axarray[i, k].tripcolor(triangulation, eigenvectors[:, j]), ax=axarray[i, k])  
    plt.setp([a.get_xticklabels() for a in axarray[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarray[:, 1]], visible=False)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Affichage des modes propres pour {} triangles".format(len(mesh[1])))
    if save_fig:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        list_img = [int(k.split('.png')[0]) for k in os.listdir(save_dir)]
        if len(list_img) == 0:
            k = 1
        else:
            k = max(list_img) + 1
        plt.savefig("{}/{}.png".format(save_dir, k))
    return eigenvalues, eigenvectors

def erreur_(solution, mesh, omega):
    x, y = mesh[0][:, 0], mesh[0][:, 1]
    triangles = mesh[1]
    K = calc_K(mesh)
    M = calc_M(mesh)
    d = (1 / (np.sqrt(2))) * np.array([1, 1])
    g = lambda x: 1/(1j*omega)*np.exp(1j * omega * np.dot(d, x))
    U_ref = np.array([g(sommet[0:2]) for sommet in mesh[0]], dtype=np.complex_)
    delta = np.subtract(U_ref, solution).real
    norm = np.dot(delta.T, np.dot(M, delta))/np.dot(U_ref.real.T, np.dot(M, U_ref.real))
    norm += np.dot(delta.T, np.dot(M, delta))/np.dot(U_ref.real.T, np.dot(M, U_ref.real))
    return norm


def plot_erreur_rayon(omega):
    erreur = list()
    plt.figure()
    y = list()
    for k in range(1, 7):
        mesh = create_mesh(read_mesh_file(k))
        points, triangle = mesh
        longueur_max = -1
        for q in range(len(mesh[1])):
            for l in range(3):
                for m in range(3):
                    longueur = np.linalg.norm(np.subtract(points[triangle[q][l]], points[triangle[q][m]]))
                    if longueur > longueur_max:
                        longueur_max = longueur
        y.append(longueur_max)
        A, F = calc_A(mesh, omega), calc_F(mesh, omega)
        U, info = spspl.cg(A,F)
        erreur.append(erreur_(U, mesh, omega))
    plt.plot([1, 2, 3, 4, 5, 6], np.log(erreur))
    plt.xlabel("Indice du maillage")
    plt.xticks([1, 2, 3, 4, 5, 6])
    plt.ylabel("Log-erreur")
    plt.title("Erreur en fonction de l'indice du maillage")
    plt.show()
    return erreur

def solve(mesh, omega=3, save_fig=False):
    A = calc_A(mesh, omega)
    F = calc_F(mesh, omega)
    U, info = spspl.cg(A,F)
    plot_triangular_solution(mesh, U.real.transpose(), omega)
    if save_fig:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        list_img = [int(k.split('.png')[0]) for k in os.listdir(save_dir)]
        if len(list_img) == 0:
            k = 1
        else:
            k = max(list_img) + 1
        plt.savefig("{}/{}.png".format(save_dir, k))
    return A, F, U

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A Python-based FEM solver for Helmholtz equation")
    parser.add_argument("mesh_index", type=int, help="Index of the mesh file to use {1, ..., 7}")
    parser.add_argument("--save_fig", type=bool, help="Flag to save figures generated", required=False)
    parser.add_argument("omega", type=float, help="Pulse of the given Helmholtz problem")
    parser.add_argument("--alpha", type=int, help="Angle of radiation", required=False, default=45)
    args = parser.parse_args()
    
    # Constants
    alpha = args.alpha
    
    print("Using mesh file n°{}".format(args.mesh_index))
    
    mesh = create_mesh(read_mesh_file(args.mesh_index))
    A, F, U = solve(mesh, omega=args.omega, save_fig=args.save_fig)
    eigenvalues, eigenvectors = solve_resonance(mesh, args.save_fig)
    plt.show()

#%%
#%%

import masterjdv3 as m
import os
import numpy as np
from mpi4py import MPI

#%%

def init_grid(n):
    grid = np.random.randint(2,size=(n,n),dtype=int)
    return grid

def read_grid(file):
    with open(file,'rb') as f:
        grid = np.load(f)
    return grid

def write_grid(file,grid):
    with open(file,'wb') as f:
        np.save(f,grid)
        

def evolution_store(grid):
    idim,jdim= grid.shape
    res_grid = np.zeros((idim,jdim),dtype=int)
    for i in range(1,idim-1):
        #specific case j=1 for initial storage values
        store0 = 0 # because store0=grid[i-1,0]+grid[i,0]+grid[i+1,0] all zeros
        store1 = grid[i - 1,1] + grid[i,1] + grid[i + 1,1] # j=1
        store2 = grid[i - 1,2] + grid[i,2] + grid[i + 1,2] # j+1=2
        nb_neigh = store0 + store1 + store2 - grid[i,1] # substract the element (i,j)
        if nb_neigh == 2:
            res_grid[i,1] = grid[i,1]
        elif nb_neigh == 3:
            res_grid[i,1] = 1
        # loop on j (we begin with j=2)
        for j in range(2,jdim - 1):
            # switch the storage values
            store0 = store1
            store1 = store2
            #compute the 3rd storage value
            store2 = grid[i - 1,j + 1] + grid[i,j + 1] + grid[i + 1,j + 1]   # 3rd subcolumn
            # compute nb_neigh
            nb_neigh = store0 + store1 + store2 - grid[i,j] # substract the element (i,j)
            if nb_neigh == 2:
                res_grid[i,j] = grid[i,j]
            elif nb_neigh == 3:
                res_grid[i,j] = 1
    return res_grid

def statalive(egrid):
    
    irange = egrid.shape[0]
    jrange = egrid.shape[1]
    
    # by default cell_state is dead
    #cell_state = 0
    nb_cell_viv = 0
    nb_cell_mort = 0
    
    # we start from 1 to idim-1
    for i in range(1,irange - 1):
        for j in range(1,jrange - 1):
    
            # loop over neighs to count living cells
            if egrid[i, j] == 1:
                nb_cell_viv = nb_cell_viv + 1
            elif egrid[i, j] == 0:
                nb_cell_mort = nb_cell_mort + 1
            nb_perc_liv = (nb_cell_viv / egrid.size) * 100
    return nb_cell_viv, nb_perc_liv

def enlarge_grid(grid):
    idim,jdim = grid.shape
    res_grid = np.zeros((idim + 2,jdim + 2))
    res_grid[1:idim + 1, 1:jdim + 1] = grid[:,:]
    return res_grid

def gamelife(grid,n):
    egrid = enlarge_grid(grid)
    print("nb_cell_viv, nb_perc_liv ::: ")
    print("")
    for iter in range(n):
        egrid = evolution_store(egrid)
        nb_cell_viv, nb_perc_liv = statalive(egrid)
        print(nb_cell_viv,nb_perc_liv)

#%% md

# New game life

#%% md

#Dans cette partie on parallelise la grille, on y retrouve alors

#%%
from mpi4py import MPI
def idim_local(grid):
    comm = MPI.COMM_WORLD
    x_dim=grid.shape[0]
    y_dim=grid.shape[1]
    processors_number=comm.size
    rank=comm.rank
    div=x_dim//processors_number
    modulo_res=x_dim%processors_number
    p_size=[div]*processors_number
    for i in range(modulo_res):
        p_size[i]+=1
    current_size = p_size[rank]
    return sum(p_size[0:rank]),current_size,y_dim


def create_local_grid(grid):
    p_sum, curr_size, y_dim = idim_local(grid)
    # np array de zeros
    enl_grid=np.zeros((curr_size+2,y_dim+2))
    enl_grid[1:curr_size+1,1:y_dim+1]=grid[p_sum:p_sum+curr_size,0:y_dim]
    return enl_grid


def new_gamelife(grid, n):
    p_sum, curr_size, y_dim = idim_local(grid)
    comm = MPI.COMM_WORLD
    p_rank = comm.rank
    nb_proc = comm.size
    egrid = create_local_grid(grid)
    for i in range(n):
        if p_rank != 0:
            # creation de la premiere ligne
            top_line = egrid[0]
            # la premiere ligne sera la ligne du dessous du traitement suivant
            comm.Recv([egrid[0, :], MPI.INT], source=p_rank - 1)
            # envoie de la premiere ligne
            comm.Send([egrid[1, :], MPI.INT], dest=p_rank - 1)

        if p_rank != nb_proc - 1:
            # create de la ligne du dessous a envoyer
            bot_line = grid[-1]
            # envoie d ela ligne du dessous
            comm.Send([egrid[curr_size, :], MPI.INT], dest=p_rank + 1)
            # la ligne du dessous sera celle du dessus pour le traitement suivant
            comm.Recv([egrid[curr_size + 1, :], MPI.INT], source=p_rank + 1)


    for iter in range(n):
        stats = statalive(egrid)
        egrid = evolution_store(egrid)
        print("cells vivantes:", stats[0], " Percentage de cells vivantes: ", stats[1])


#%%
file = os.path.join("", "jdv_1000.grid")
grid = read_grid(file)
grid = init_grid(grid.shape[0])

new_gamelife(grid,10)

#%%



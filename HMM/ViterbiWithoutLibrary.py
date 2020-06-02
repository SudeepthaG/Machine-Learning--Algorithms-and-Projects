

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


def main():
    # Reading input data
    with open("hmm-data.txt") as input:
        for i in range(2):
            next(input)
        # Reading grid-world
        grid=[]
        for i in range(10):
            line=input.readline()
            line=list(map(int,line.split()))
            grid.append(line)
        # print(grid)
        for i in range(4):
            next(input)

        # Reading tower locations
        towers=[]
        for i in range(4):
            line = input.readline()
            line=line.split(": ")[1]
            line = list(map(int, line.split()))
            towers.append(line)
        # print(towers)
        for i in range(4):
            next(input)

        #Reading Noisy Distances to Towers 1, 2, 3 and 4 Respectively for 11 Time-Steps
        noisy_dist = []
        for i in range(11):
            line = input.readline()
            line = list(map(float, line.split()))
            noisy_dist.append(line)
        # print(noisy_dist)
    transitionMatrix=calculateTransitionProbabilityMatrix(grid)
    emissionMatrix=calculateEmissionProbabilityMatrix(towers,noisy_dist)
    path=viterbiImplementation(grid, transitionMatrix, emissionMatrix)
    print(path)


def calculateTransitionProbabilityMatrix(grid):
    grid = np.array(grid)
    transitionMatrix=np.zeros((100,100))
    coordinates=[]
    for x in range(10):
        for y in range(10):
            xy_coordinate=[x,y]
            coordinates.append(xy_coordinate)
    coordinates=np.array(coordinates)
    # print(coordinates)
    joined_coordinates=np.arange(100).reshape((10,10))
    # print(joined_coordinates)
    for i in range(100):
        possible_moves=[[coordinates[i][0]+1,coordinates[i][1]],[coordinates[i][0],coordinates[i][1]+1],[coordinates[i][0]-1,coordinates[i][1]],[coordinates[i][0],coordinates[i][1]-1]]
        # print(possible_moves)
        if(grid[coordinates[i][0],coordinates[i][1]]!=0):
            valid_moves=[]
            for x,y in possible_moves:
                if((x>=0 and y>=0 and x<=9 and y<=9 and grid[x,y]==1)):
                    valid_moves.append({"x":x,"y":y})
            if(len(valid_moves)>0):
                for move in valid_moves:
                    transitionMatrix[i,joined_coordinates[move["x"],move["y"]]]=1.0/len(valid_moves)
    # print(transitionMatrix)
    return transitionMatrix


def calculateEmissionProbabilityMatrix(towers, noisy_dist):
    coordinates = []
    for x in range(10):
        for y in range(10):
            xy_coordinate = [x, y]
            coordinates.append(xy_coordinate)
    coordinates = np.array(coordinates)
    noisy_dist = np.array(noisy_dist)
    emissionMatrix=np.ones((100,11))
    for index, tower in enumerate(towers):
        observations = noisy_dist[:, index].tolist()
        # print(observations)
        tempEmission = np.zeros((100, 11))
        for index1, observation in enumerate(observations):
            for i in range(100):
                distance = np.linalg.norm(np.array(tower) - [coordinates[i]])
                if (distance == 0):
                    if (observation == 0):
                        tempEmission[i, index1] = (1+0.1) / (1 + 100 * 0.1)
                        continue
                    else:
                        tempEmission[i, index1] = 0.1 / (1 + 100 * 0.1)
                        continue
                else:
                    maxDist = np.floor(distance * 1.3 * 10)
                    minDist = np.ceil(distance * 0.7 * 10)
                    valid_range_length = maxDist - minDist + 1
                    probability = 1.0 / valid_range_length
                    if (observation * 10 >= minDist and observation * 10 <= maxDist):
                        tempEmission[i,index1] = probability
        emissionMatrix *= tempEmission
    # print(emissionMatrix)
    return emissionMatrix


def viterbiImplementation(grid, transitionMatrix, emissionMatrix):
    grid=np.array(grid)
    empty_cells = np.count_nonzero(grid == 1)
    new_grid = np.copy(grid).astype(float)
    new_grid[new_grid > 0.0] = 1.0 / empty_cells
    # print(new_grid)
    coordinates = []
    for x in range(10):
        for y in range(10):
            xy_coordinate = [x, y]
            coordinates.append(xy_coordinate)
    coordinates = np.array(coordinates)
    temp = np.zeros((100, 11))
    temp1 = np.zeros((100, 11))
    for i in range(100):
        temp[:, 0] = new_grid.ravel() * emissionMatrix[:, 0]
    for j in range(1, 11):
        for i in range(100):
            column = temp[:, j - 1] * transitionMatrix[:, i] * emissionMatrix[:, j]
            temp[i, j] = np.max(column)
            temp1[i, j] = np.argmax(column)
    temp_path = np.zeros((11,), dtype=int)
    path = np.zeros((11, 2), dtype=int)
    temp_path[10] = np.argmax(temp[:, 10])
    path[10] = coordinates[temp_path[10]]
    for i in range(10, 0, -1):
        temp_path[i - 1] = temp1[temp_path[i], i]
        path[i - 1] = coordinates[temp_path[i - 1]]
        # print(path)
    # print(path)
    return path


if __name__ == "__main__":
    main()

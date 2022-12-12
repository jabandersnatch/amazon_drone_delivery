import numpy as np
import random

array_a = [np.array([1, 0]), np.array([2, 3, 4, 5])]

array_b = [np.array([2, 3, 4, 0]), np.array([1, 5])]

def comb(a, b):
    '''
    This function will combine the two arrays such a
    number will never be repeated.
    a and b have the same length
    '''
    def check_if_repeated(array, element):
        '''
        This function will check if an element is repeated in an array
        '''
        if element in array:
            return True
        else:
            return False

    # First remove the last element of all the arrays and store them in a list
    last_a = [i[-1] for i in a]
    last_b = [i[-1] for i in b]
    # Now remove the last element from all the arrays
    a = [i[:-1] for i in a]
    b = [i[:-1] for i in b]
    print ('End warehouses: ', last_a, last_b)
    # Then we combine last_a and last_b  as a matrix where the rows are the
    # elements of last_a and the columns are the elements of last_b
    comb = np.array([last_a, last_b])
    print ('Combination matrix: ', comb)
    rows_visited = []
    nodes_visited = []

    # we will create a new matrix with empty arrays the same size as the number of rows
    # in the a and b arrays
    new_matrix = [np.array([]) for i in range(len(a))]


    # We will now iterate over the rows of a and b
    # We will start by selecting a random row 
    while len(rows_visited) < len(a):
        # We select a random row
        row = np.random.randint(0, len(a))
        # If the row has already been visited, we select another one
        if row in rows_visited:
            continue
        # We add the row to the list of visited rows
        rows_visited.append(row)
        # We print the current row
        print ('Drone {}' .format(row))
        # Now we merge the two arrays such as
        # merge = [a1, b1, a2, b2, ...]
        # note that a[row] and b[row] have different length
        merge = []
        # First get the length of the shortest array
        min_len = min(len(a[row]), len(b[row]))

        # Add length of the rows and get the mean of the two and floor it

        # This will be the number of elements we will add to merge
        # from a[row] and b[row]
        mean = int(np.floor((len(a[row]) + len(b[row])) / 2))

        # Now we iterate over the shortest array
        for i in range(min_len):
            # Make a choice between a[row] and b[row]
            choice = np.random.randint(0, 2)
            # If choice is 0, we add an element from a[row]
            if choice == 0:
                if check_if_repeated(merge, a[row][i]):
                    #Check if the element is in nodes_visited
                    if check_if_repeated(nodes_visited, a[row][i]):
                        continue
                    else:
                        merge.append(b[row][i])
                else:
                    # Check if the element is in nodes_visited
                    if check_if_repeated(nodes_visited, a[row][i]):
                        continue
                    else:
                        merge.append(a[row][i])
            # If choice is 1, we add an element from b[row]
            else:
                if check_if_repeated(merge, b[row][i]):
                    # Check if the element is in nodes_visited
                    if check_if_repeated(nodes_visited, b[row][i]):
                        continue
                    else:
                        merge.append(a[row][i])
                else:
                    # Check if the element is in nodes_visited
                    if check_if_repeated(nodes_visited, b[row][i]):
                        continue
                    else:
                        merge.append(b[row][i])

        # Now we add the remaining elements of the longest array
        # to merge ensuring that the elements are not repeated
        if len(a[row]) > len(b[row]):
            for i in range(len(a[row]) - min_len):
                if check_if_repeated(merge, a[row][i + min_len]):
                    # Check if the element is in nodes_visited
                    if check_if_repeated(nodes_visited, a[row][i + min_len]):
                        continue
                    else:
                        merge.append(a[row][i + min_len])
                else:
                    # Check if the element is in nodes_visited
                    if check_if_repeated(nodes_visited, a[row][i + min_len]):
                        continue
                    else:
                        merge.append(a[row][i + min_len])

        # Now we make a choice between cuting the last element of
        # merge or the first element of merge in order to make sure
        # that the size of merge is equal to mean
        choice = np.random.randint(0, 2)
        elements_to_remove = len(merge) - mean
        if choice == 0:
            merge = merge[:-elements_to_remove]
        else:
            merge = merge[elements_to_remove:]

        # Now we print the merged array
        print ('Merge: ', merge)
        # Now we add the last element of merge to the list of visited nodes
        for i in merge:
            nodes_visited.append(i)
        # Now we add to the merge row a random choice between the last_a and last_b
        choice = np.random.randint(0, 2)
        merge.append(comb[choice, row])
        # Now we print the final merge
        print ('Final merge: ', merge)

        # Now we add the merge to the new new_matrix in order to return it
        new_matrix[row] = np.array(merge)
        
    # print nodes_visited
    print ('Nodes visited: ', nodes_visited)

    # Now we check that all the nodes in a and b have been visited
        
    # First we get the list of all the nodes in a and b
    all_nodes = []
    for i in a:
        all_nodes.extend(i)
    for i in b:
        all_nodes.extend(i)

    # Now we delete the repeated nodes
    all_nodes = list(set(all_nodes))

    # Now we check if all the nodes have been visited

    if set(all_nodes) == set(nodes_visited):
        print ('All nodes have been visited')

    # Now we make a list of the nodes that have not been visited
    nodes_not_visited = list(set(all_nodes) - set(nodes_visited))
    print ('Nodes not visited: ', nodes_not_visited)

    # Now we add the nodes that have not been visited to the new matrix in a random order
    # First we shuffle the nodes_not_visited
    np.random.shuffle(nodes_not_visited)

    # Now we add the nodes to the new new_matrix
    for i in range(len(nodes_not_visited)):

        # We select a random row
        row = np.random.randint(0, len(a))
        # First we check that size of the row is not one
        if len(new_matrix[row]) == 1:
            # If it is one we add the not visitted nodes at the start of the row
            new_matrix[row] = np.insert(new_matrix[row], 0, nodes_not_visited[i])
        
        else:
            # then we add the node to the row at a random position in the row
            # all the postions are valid except the last one
            pos = np.random.randint(0, len(new_matrix[row]) - 1)
            new_matrix[row] = np.insert(new_matrix[row], pos, nodes_not_visited[i])

        # turn the new_matrix into a list of lists
        for i in range(len(new_matrix)):
            new_matrix[i] = new_matrix[i].tolist()

        # Now we return the new_matrix
        return new_matrix

def mutation(matrix):
    '''
    Mutate the matrix
    The mutation is done by changing the position of two elements in the matrix the matrix 
    cant have the same element twice
    shuffle some the nodes between rows
    '''
    new_matrix = []

    # First remove the last element of each row
    # and store it in a list

    last_elements = [row[-1] for row in matrix]
    matrix = [row[:-1] for row in matrix]

    for i in range(len(matrix)):
        new_matrix.append(matrix[i])

    # Now we select two random rows

    row1 = np.random.randint(0, len(matrix))
    row2 = np.random.randint(0, len(matrix))

    # Now we select two random positions in the rows
    pos1 = np.random.randint(0, len(matrix[row1]))
    pos2 = np.random.randint(0, len(matrix[row2]))

    # Now we swap the elements in the rows
    new_matrix[row1][pos1], new_matrix[row2][pos2] = new_matrix[row2][pos2], new_matrix[row1][pos1]

    # Now we add the last elements to the rows
    new_matrix[row1].append(last_elements[row1])

    # Now we return the new_matrix

    return new_matrix



        
print ('Array a: ', array_a)
print ('Array b: ', array_b)
new_matrix=comb(array_a, array_b)

print ('New matrix: ', new_matrix)

# Now we test the mutation function
print ('Mutation: ', mutation(new_matrix))


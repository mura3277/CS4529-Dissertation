def test(x):
    y = 0
    for i in range(x):
        y += i
    return y

#def format(rays, idx):
#    new = zeros((len(idx[-1][0]), 3))
#    for c in range(len(idx[-1][0])):
#        new[c] = rays[:, idx[-1][0][c], idx[-1][1][c]]
#    return new
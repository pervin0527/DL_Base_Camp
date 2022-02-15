global_var1 = 1
global_var2 = "GLOBAL"

def test():
    local_var1 = 100
    local_var2 = "LOCAL"
    print(locals())

class test:
    class_var1 = 200
    class_var2 = "CLASS_VAR"

print(globals())
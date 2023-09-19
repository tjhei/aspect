import subprocess
import numpy as np
import matplotlib.pyplot as plt

ranks = 2
release_mode = True
build_dir = "./../../build/"


def send(process, message):
    print(">", message)
    process.stdin.write((message+"\n").encode())
    process.stdin.flush()
    
def receive(process):
    line = process.stdout.readline().decode()
    line = line[:-1] # strip endline
    print("|", line)

    return line

def get_residual(process):
    value = 0.
    while True:
        line = receive(process)
        if line.startswith("     Velocity"):
            word = line.split()
            value = float(word[-1])
            return value

def wait_for_prompt(process):
    while True:
        line = receive(process)
        if line=="?":
            return

def create_temp_wb_file (input_wb_file, new_wb_parameter_value, old_wb_parameter_value):
    with open(input_wb_file, 'r') as file:
        filedata = file.read()
    
    filedata = filedata.replace('"thickness"' + ':[' + old_wb_parameter_value + ']', 
                                '"thickness"' + ':[' + str(new_wb_parameter_value) + ']')
    
    with open ('temp.wb', 'w') as file2:
        file2.write(filedata)

    file.close()
    file2.close()    

binary = build_dir + "aspect-release" if release_mode else "aspect"
process =  subprocess.Popen(["./../../build/aspect-release", "1.prm"], 
                        stdin = subprocess.PIPE,
                        stdout = subprocess.PIPE)

# define the list of values we want to test here
test_thickness_values = np.arange(50e3, 200e3, 10e3)
rms_residual = []

def run_parameter_search ():

    for i in range (len(test_thickness_values)):
        wait_for_prompt(process)
        create_temp_wb_file('simple.wb', test_thickness_values[i], "195e3")
    
        send(process, "wb temp.wb")
        wait_for_prompt(process)
        
        send(process, "continue")
        residual = get_residual(process)
        rms_residual.append(residual)

        

run_parameter_search()
    
plt.plot(test_thickness_values, rms_residual)
plt.xlabel("Fault thickness")
plt.ylabel("Rms velocity residual")
print (test_thickness_values, rms_residual)
plt.show()
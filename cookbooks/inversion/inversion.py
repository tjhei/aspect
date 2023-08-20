import subprocess

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

def wait_for_prompt(process):
    while True:
        line = receive(process)
        if line=="?":
            return

binary = build_dir + "aspect-release" if release_mode else "aspect"
process =  subprocess.Popen(["mpirun", "-n", str(ranks), "./../../build/aspect-release", "1.prm"], 
                        stdin = subprocess.PIPE,
                        stdout = subprocess.PIPE)

wait_for_prompt(process)

send(process, "wb temp.wb")
wait_for_prompt(process)
send(process, "continue")
wait_for_prompt(process)

send(process, "wb temp.wb")
wait_for_prompt(process)
send(process, "thermal-exp 1e-5")
wait_for_prompt(process)
send(process, "continue")
wait_for_prompt(process)

send(process, "wb simple.wb")
wait_for_prompt(process)
send(process, "continue")
wait_for_prompt(process)

send(process, "continue")
wait_for_prompt(process)
send(process, "quit")

while True:
    line = receive(process)
    if line=="?": break

    

if False:
    receive(process) # header
    line = receive(process) # prompt
    assert(line == "?")
    send(process, "help")
    receive(process) # header
    commands = receive(process) # reply
    line = receive(process) # prompt
    assert(line == "?")

    #send(process, "what")

    send(process, "quit")

    line = receive(process)
    assert(line =="goodbye.")

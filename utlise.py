import os

class log_print:
    def __init__(self,logfile=None,save_log=True):
        self.logfile = logfile
        self.save_log = save_log

    def __call__(self, string=None, end='\n'):
        string = str(string)
        if self.save_log==True:
            with open(self.logfile,'a') as f:
                f.write(string+end)
        print(string,end=end)


def create_checkpoint():
    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")
    exp_num = len(os.listdir("./checkpoint"))
    os.mkdir(f"./checkpoint/exp{exp_num}")
    return f"./checkpoint/exp{exp_num}/"
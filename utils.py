from os.path import join as ospj
import sys
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class Logger(object):
    def __init__(self):
        self._terminal = sys.stdout
        self._timestr = datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S")

    def create_log(self, log_path):
        self.log = open(log_path + "/log-%s.txt" % self._timestr, "a", 1)

    def write(self, message):
        self._terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class Recorder:
    def __init__(self, larger_is_better=True):
        self.history = []
        self.larger_is_better = larger_is_better
        self.best_at = None
        self.best_val = None

    def is_better_than(self, x, y):
        if self.larger_is_better:
            return x > y
        else:
            return x < y

    def update(self, val):
        self.history.append(val)
        if len(self.history) == 1 or self.is_better_than(val, self.best_val):
            self.best_val = val
            self.best_at = len(self.history) - 1

    def is_current_best(self):
        return self.best_at == len(self.history) - 1


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.history.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_average_meters(n):
    return [AverageMeter() for _ in range(n)]


def get_timestr(exp_dirname):
    timestr = exp_dirname.split("g")[1].split("_")[0]
    return timestr


def write_cmd_to_file(log_dir, argv):
    with open(ospj(log_dir, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(argv))


def get_exp_dir():
    import socket
    host_name = socket.gethostname()
    EXP_DIR = "exps_clf"
    if "whisman" in host_name:
        return '/home/meng/mit/%s/'%(EXP_DIR)
    elif "node0" in host_name or "service000" in host_name:
        return '/nobackup/users/mengyue/%s/'%(EXP_DIR)
    elif "realm" in host_name:
        return '/home/meng/%s/'%(EXP_DIR)
    else:
        exit("unrecognized hostname: %s - you can set exp root dir by modifying --exp_root_dir" % (host_name))


def viz_graph(prev_s, curr_s, dbg_t, viz_path):
    # # TODO Debug only
    # dbg_len = 300
    # s = s_list
    # s = torch.from_numpy(s).float().cuda()
    # prev_s = None
    # for dbg_t in range(dbg_len):
    #     u = actor(s)
    #     s_next = dynamics(s, u, args)
    #     prev_s = s
    #     s = s_next

    # plot previous
    prev_s = prev_s.detach().cpu().numpy()
    th = np.arctan2(prev_s[:, 1], prev_s[:, 0])
    thdot = prev_s[:, 2]
    plt.figure(figsize=(6, 6))
    plt.scatter(th, thdot)
    plt.axis('scaled')
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.ylim(-8.0, 8.0)
    plt.savefig(ospj(viz_path,"dbg_%03d.png"%(dbg_t)))
    plt.close()

    # plot vector field
    curr_s = curr_s.detach().cpu().numpy()
    th1 = np.arctan2(curr_s[:, 1], curr_s[:, 0])
    thdot1 = curr_s[:, 2]

    plt.figure(figsize=(6, 6))
    th_vec = th1 - th
    thdot_vec = thdot1 - thdot
    C = np.hypot(th_vec, thdot_vec)
    th_norm = th_vec / C
    thdot_norm = thdot_vec / C

    Q = plt.quiver(th, thdot, th_norm, thdot_norm, C, units='xy', cmap=cm.gnuplot)
    plt.colorbar()

    plt.savefig(ospj(viz_path, "vec_%03d.png" % (dbg_t)))
    plt.close()
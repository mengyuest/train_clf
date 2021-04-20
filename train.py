import torch
from torch import nn
import gym
import argparse
import numpy as np
import os, sys, time
from os.path import join as ospj
import utils
from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib import cm
import test_rl
from gym import wrappers


class CLF(nn.Module):
    def __init__(self, args):
        super(CLF, self).__init__()
        self.args = args
        self.linear_list = nn.ModuleList()
        input_dim = 3  # cos, sin, thdot,
        self.linear_list.append(nn.Linear(input_dim, args.clf_hiddens[0]))
        for i, hidden in enumerate(args.clf_hiddens):
            if i == len(args.clf_hiddens) - 1:
                self.linear_list.append(nn.Linear(args.clf_hiddens[i], 1))
            else:
                self.linear_list.append(nn.Linear(args.clf_hiddens[i], args.clf_hiddens[i + 1]))

    def forward(self, x):
        for i, hidden in enumerate(self.args.clf_hiddens):
            x = nn.Tanh()(self.linear_list[i](x))
        x = self.linear_list[len(self.args.clf_hiddens)](x)
        x = nn.Sigmoid()(x) * self.args.clf_scale
        return x


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        self.linear_list = nn.ModuleList()
        input_dim = 3  # cos, sin, thdot,
        self.linear_list.append(nn.Linear(input_dim, args.actor_hiddens[0]))
        for i, hidden in enumerate(args.actor_hiddens):
            if i == len(args.actor_hiddens) - 1:
                self.linear_list.append(nn.Linear(args.actor_hiddens[i], 1))
            else:
                self.linear_list.append(nn.Linear(args.actor_hiddens[i], args.actor_hiddens[i + 1]))

    def forward(self, x):
        for i, hidden in enumerate(self.args.actor_hiddens):
            x = nn.ReLU()(self.linear_list[i](x))
        x = self.linear_list[len(self.args.actor_hiddens)](x)
        x = nn.Tanh()(x) * self.args.max_torque
        return x


def get_hyperparams():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root_dir', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default="exp")
    parser.add_argument('--random_seed', type=int, default=1007)
    parser.add_argument('--num_epochs', type=int, default=50000)
    parser.add_argument('--print_freq', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--viz_freq', type=int, default=1000)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sim_len', type=int, default=50)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--max_torque', type=float, default=2.0)
    parser.add_argument('--max_speed', type=float, default=8.0)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--fix_init', action='store_true', default=False)
    parser.add_argument('--random_sampled', action='store_true', default=False)
    parser.add_argument('--sample_only_once', action='store_true', default=False)

    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--gpus', type=str, default=None)

    parser.add_argument('--clf_hiddens', type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument('--clf_scale', type=float, default=0.3)
    parser.add_argument('--preset_clf', action='store_true', default=False)
    parser.add_argument('--clf_pretrained_path', type=str, default=None)

    parser.add_argument('--actor_hiddens', type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument('--preset_actor', action='store_true', default=False)
    parser.add_argument('--actor_pretrained_path', type=str, default="./rl/models/baseline.pth")

    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--margin_grad', type=float, default=0.02)
    parser.add_argument('--weight_zero', type=float, default=1.0)
    parser.add_argument('--weight_range', type=float, default=1.0)
    parser.add_argument('--weight_grad', type=float, default=1.0)

    parser.add_argument('--debug_viz', action='store_true', default=False)
    parser.add_argument('--debug_viz_len', type=int, default=10)
    parser.add_argument('--eval_render', action='store_true', default=False)
    parser.add_argument('--eval_render_len', type=int, default=200)
    return parser.parse_args()


def set_random_seed(env, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)


def dynamics(s, u, args):
    # constants
    g = 10.0
    m = 1.
    l = 1.
    dt = args.dt

    # parse input
    costh, sinth, thdot = s.split([1, 1, 1], dim=-1)
    th = torch.atan2(sinth, costh)
    u = torch.clamp(u, -args.max_torque, args.max_torque)

    # dynamics
    newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
    newth = th + newthdot * dt
    newthdot = torch.clamp(newthdot, -args.max_speed, args.max_speed)
    s_next = torch.cat([torch.cos(newth), torch.sin(newth), newthdot], dim=-1)
    return s_next


def main():
    t0 = time.time()
    args = get_hyperparams()

    # environment
    env = gym.make('Pendulum-v0')
    # env = wrappers.Monitor(env, './tmp/', force=True)
    assert env.max_torque == args.max_torque
    assert env.max_speed == args.max_speed
    set_random_seed(env, args.random_seed)

    # models
    clf = CLF(args)
    if args.preset_clf:
        m = torch.load(args.clf_pretrained_path)
        clf.load_state_dict(m)

    if args.preset_actor:
        actor = test_rl.Actor()
        actor.load(args.actor_pretrained_path)
    else:
        actor = Actor(args)

    s_null = torch.tensor([[1.0, 0.0, 0.0]])  # torch.from_numpy(np.array([[1, 0, 0]])).float()

    if args.use_cuda:
        clf = clf.cuda()
        actor = actor.cuda()
        s_null = s_null.cuda()

    # optimizer
    if args.preset_actor:
        optimizer = torch.optim.SGD(clf.parameters(), args.lr)
    elif args.preset_clf:
        optimizer = torch.optim.SGD(actor.parameters(), args.lr)
    else:
        optimizer = torch.optim.SGD(
            list(clf.parameters()) + list(actor.parameters()), args.lr)

    # watchlist
    losses, rewards, losses_zero, losses_range, losses_grad = utils.get_average_meters(5)

    if args.fix_init:
        env.reset()
        assert args.batch_size == 1

    # setup directory and logs
    exp_root = utils.get_exp_dir() if args.exp_root_dir is None else args.exp_root_dir
    logger = utils.Logger()
    exp_dirname = "g%s_%s" % (logger._timestr, args.exp_name)
    exp_fullname = ospj(exp_root, exp_dirname)
    model_path = ospj(exp_fullname, "models")
    viz_path = ospj(exp_fullname, "viz")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(viz_path, exist_ok=True)
    logger.create_log(exp_fullname)
    sys.stdout = logger
    writer = SummaryWriter(exp_fullname)
    utils.write_cmd_to_file(exp_fullname, sys.argv)
    copyfile("./train.py", ospj(model_path, "train.py"))
    copyfile("./utils.py", ospj(model_path, "utils.py"))

    s_list = []

    # TRAIN LOOP
    for ep_i in range(args.num_epochs):
        score = 0
        # random state-space sampling
        if args.random_sampled:
            if args.sample_only_once == False or ep_i == 0:
                N = args.batch_size * args.sim_len
                rth = np.random.random(N) * np.pi * 2
                rthdot = np.random.random(N) * 16 - 8
                s_list = np.stack((np.cos(rth), np.sin(rth), rthdot), axis=-1)

                if args.debug_viz:
                    curr_s = torch.from_numpy(s_list).float()
                    if args.use_cuda:
                        curr_s = curr_s.cuda()
                    for dbg_t in range(args.debug_viz_len):
                        u = actor(curr_s)
                        prev_s = curr_s
                        curr_s = dynamics(curr_s, u, args)
                        utils.viz_graph(prev_s, curr_s, dbg_t, viz_path)
            s = torch.from_numpy(s_list).float()
            if args.use_cuda:
                s = s.cuda()
            u = actor(s)

        else:  # trajectory-wise sampling
            s_list = []
            u_list = []
            for b_i in range(args.batch_size):
                if args.fix_init:
                    env.state[0] = 0
                    env.state[1] = 0
                    env.last_u = None
                    state = np.array([1, 0, 0])
                else:
                    state = env.reset()

                state = torch.from_numpy(state).float().unsqueeze(0)
                if args.use_cuda:
                    state = state.cuda()

                for t in range(args.sim_len):
                    u = actor(state).squeeze(-1)
                    state_, reward, done, _ = env.step(u.detach().cpu().numpy())
                    state_ = torch.from_numpy(state_).float().unsqueeze(0)
                    if args.use_cuda:
                        state_ = state_.cuda()
                    score += reward
                    if args.render:
                        env.render()

                    s_list.append(state)
                    u_list.append(u)
                    state = state_
            s = torch.cat(s_list, dim=0)  # (batch_size * sim_len, 3)
            u = torch.stack(u_list, dim=0)  # (batch_size * sim_len, 1)

        # forward
        v = clf(s)  # (batch_size * sim_len, 1)
        v_null = clf(s_null)
        s_next = dynamics(s, u, args)
        v_next = clf(s_next)
        v_dot = (v_next - v) / args.dt

        # find values that outside the unit circle
        th_0 = torch.atan2(s[:, 1], s[:, 0])
        thd_0 = s[:, 2]
        v_out = v[torch.where(th_0 ** 2 + thd_0 ** 2 > 1)]

        # losses
        loss_zero = v_null ** 2  # v0=0
        loss_range = torch.mean(nn.ReLU()(0.1 - v_out))  # outside the unit circle, value >= 0.1
        loss_grad = torch.mean(torch.relu(args.margin_grad + v_dot))  # vdot<=-margin_grad

        loss_zero = args.weight_zero * loss_zero
        loss_range = args.weight_range * loss_range
        loss_grad = args.weight_grad * loss_grad

        loss = loss_zero + loss_range + loss_grad

        # recording
        rewards.update(score / args.batch_size)
        losses.update(loss.detach().cpu().item())
        losses_zero.update(loss_zero.detach().cpu().item())
        losses_range.update(loss_range.detach().cpu().item())
        losses_grad.update(loss_grad.detach().cpu().item())
        writer.add_scalar("0_reward", score, ep_i)
        writer.add_scalar("1_loss", loss, ep_i)
        writer.add_scalar("2_loss_zero", loss_zero, ep_i)
        writer.add_scalar("3_loss_range", loss_range, ep_i)
        writer.add_scalar("4_loss_grad", loss_grad, ep_i)

        # back-propagation
        loss.backward()
        optimizer.step()

        # print
        if ep_i % args.print_freq == 0:
            print("[%05d/%05d] reward:%.4f(%.4f) loss:%.4f(%.4f) zero:%.4f(%.4f) range:%.4f(%.4f) grad:%.4f(%.4f)" % (
                ep_i, args.num_epochs, rewards.val, rewards.avg, losses.val, losses.avg,
                losses_zero.val, losses_zero.avg, losses_range.val, losses_range.avg, losses_grad.val, losses_grad.avg,
            ))

        # visualization
        if ep_i % args.viz_freq == 0:
            clf.eval()
            # visualize lyapunov function over th-thdot plane
            n_ths = 63
            n_thdots = 161
            ths = np.linspace(-np.pi, np.pi, n_ths)
            thdots = np.linspace(-8, 8, n_thdots)
            thv, thdotsv = np.meshgrid(ths, thdots, indexing="ij")
            thv = torch.from_numpy(thv).float()
            thdotsv = torch.from_numpy(thdotsv).float()
            if args.use_cuda:
                thv = thv.cuda()
                thdotsv = thdotsv.cuda()
            inputv = torch.stack([torch.cos(thv.flatten()), torch.sin(thv.flatten()), thdotsv.flatten()], dim=-1)
            outputv = clf(inputv)
            plt.imshow(outputv.detach().cpu().numpy()[:, 0].reshape((n_ths, n_thdots)).T, origin='lower',
                       cmap=cm.inferno)
            plt.colorbar()
            plt.savefig(ospj(viz_path, "2_e%05d_heat.png" % (ep_i)), bbox_inches='tight', pad_inches=0)
            plt.close()

            # test simulation
            if args.eval_render:
                s = env.reset()
                for dbg_t in range(args.eval_render_len):
                    action = actor(torch.from_numpy(s).unsqueeze(0).float().cuda())
                    s, reward, _, _ = env.step(action.detach().cpu().squeeze(0).numpy())
                    env.render()

            clf.train()

        # save model
        if ep_i % args.save_freq == 0:
            torch.save(clf.state_dict(), "%s/clf_%05d.ckpt" % (model_path, ep_i))
            torch.save(actor.state_dict(), "%s/actor_%05d.ckpt" % (model_path, ep_i))
        writer.flush()

    env.close()
    t1 = time.time()
    print("Finished in %.4f seconds" % (t1 - t0))


if __name__ == "__main__":
    main()

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
from gym import spaces

class CLF(nn.Module):
    def __init__(self, args):
        super(CLF, self).__init__()
        # self.relu = nn.ReLU()
        self.relu = nn.Tanh()
        self.args = args
        self.linear_list = nn.ModuleList()
        input_dim = 3  # cos, sin, thdot,
        self.linear_list.append(nn.Linear(input_dim, args.clf_hiddens[0]))
        for i,hidden in enumerate(args.clf_hiddens):
            if i==len(args.clf_hiddens)-1:
                self.linear_list.append(nn.Linear(args.clf_hiddens[i], 1))
            else:
                self.linear_list.append(nn.Linear(args.clf_hiddens[i], args.clf_hiddens[i + 1]))

    def forward(self, x):
        if self.args.preset_clf and self.args.preset_clf_shape:
            cth, sth, thdot = x.split([1,1,1], dim=-1)
            th = torch.atan2(sth, cth)
            return (th**2+thdot**2)**0.5
        else:
            # th, thdot = x.split([1,1], dim=-1)
            # x = torch.cat([torch.cos(th), torch.sin(th), thdot], dim=-1)

            for i, hidden in enumerate(self.args.clf_hiddens):
                x = self.relu(self.linear_list[i](x))
            x = self.linear_list[len(self.args.clf_hiddens)](x)

            x = nn.Sigmoid()(x) * 0.3

            return x


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.relu = nn.ReLU()
        self.args = args
        self.linear_list = nn.ModuleList()
        input_dim = 3  # cos, sin, thdot,
        self.linear_list.append(nn.Linear(input_dim, args.policy_hiddens[0]))
        for i, hidden in enumerate(args.policy_hiddens):
            if i == len(args.policy_hiddens) - 1:
                self.linear_list.append(nn.Linear(args.policy_hiddens[i], 1))
            else:
                self.linear_list.append(nn.Linear(args.policy_hiddens[i], args.policy_hiddens[i + 1]))

    def forward(self, x):
        # th, thdot = x.split([1, 1], dim=-1)
        # x = torch.cat([torch.cos(th), torch.sin(th), thdot], dim=-1)

        for i, hidden in enumerate(self.args.policy_hiddens):
            x = self.relu(self.linear_list[i](x))
        x = self.linear_list[len(self.args.policy_hiddens)](x)
        #x = nn.Sigmoid()(x) * self.args.max_torque*2 - self.args.max_torque
        x = nn.Tanh()(x) * self.args.max_torque
        return x

def get_hyperparams():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--viz_freq', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=1007)
    parser.add_argument('--exp_name', type=str, default="exp")
    parser.add_argument('--pretrained_path', type=str, default=None)  # TODO
    parser.add_argument('--clf_hiddens', type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument('--policy_hiddens', type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument('--eval_mode', action='store_true', default=False)
    parser.add_argument('--debug_mode', action='store_true', default=False)
    parser.add_argument('--sim_len', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--margin_pos', type=float, default=0.001)
    parser.add_argument('--margin_grad', type=float, default=0.001)
    parser.add_argument('--weight_zero', type=float, default=1.0)
    parser.add_argument('--weight_pos', type=float, default=1.0)
    parser.add_argument('--weight_grad', type=float, default=1.0)
    parser.add_argument('--weight_bound', type=float, default=1.0)
    parser.add_argument('--margin_factor', type=float, default=0.1)
    parser.add_argument('--fix_init', action='store_true', default=False)
    parser.add_argument('--random_sampled', action='store_true', default=False)
    parser.add_argument('--sample_only_once', action='store_true', default=False)
    parser.add_argument('--use_cuda', action='store_true', default=False)

    parser.add_argument('--preset_clf', action='store_true', default=False)
    parser.add_argument('--preset_clf_shape', action='store_true', default=False)
    parser.add_argument('--clf_pretrained_path', type=str, default="../DeepReinforcementLearning/models/1450.ckpt")

    parser.add_argument('--max_torque', type=float, default=2.0)
    parser.add_argument('--max_speed', type=float, default=8.0)

    parser.add_argument('--preset_actor', action='store_true', default=False)
    parser.add_argument('--actor_pretrained_path', type=str, default="../DeepReinforcementLearning/models/1450.ckpt")

    return parser.parse_args()

# python train.py --gpus 0 --exp_name debug

def set_random_seed(env, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

def dynamics(s, u, args):
    costh, sinth, thdot = s.split([1, 1, 1], dim=-1)
    th = torch.atan2(sinth, costh)
    g = 10.0
    m = 1.
    l = 1.
    dt = args.dt
    max_torque = args.max_torque
    max_speed = args.max_speed

    u = torch.clamp(u, -max_torque, max_torque)

    newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th+np.pi) + 3. / (m * l ** 2) * u) * dt
    newth = th + newthdot * dt
    newthdot = torch.clamp(newthdot, -max_speed, max_speed)

    s_next = torch.cat([torch.cos(newth), torch.sin(newth), newthdot], dim=-1)
    return s_next

def main():
    t0 = time.time()
    args = get_hyperparams()

    # environment
    env = gym.make('Pendulum-v0')
    set_random_seed(env, args.random_seed)

    env.max_torque=args.max_torque
    env.max_speed=args.max_speed
    # env.action_space = spaces.Box(
    #         low=-args.max_torque,
    #         high=args.max_torque, shape=(1,),
    #         dtype=np.float32
    #     )


    # models
    clf = CLF(args)
    if args.preset_clf:
        m = torch.load(args.clf_pretrained_path)
        clf.load_state_dict(m)

    if args.preset_actor:
        import test_rl
        actor = test_rl.Actor()
        actor.load(args.actor_pretrained_path)
    else:
        actor = Actor(args)


    if args.use_cuda:
        clf = clf.cuda()
        actor = actor.cuda()

    # setup directory and logs
    exp_root = utils.get_exp_dir()
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

    # write cmd line
    utils.write_cmd_to_file(exp_fullname, sys.argv)

    # copy code to inside
    copyfile("./train.py", ospj(model_path, "train.py"))
    copyfile("./utils.py", ospj(model_path, "utils.py"))

    if args.preset_actor:

        optimizer = torch.optim.SGD(clf.parameters(), args.lr)
    elif args.preset_clf:
        optimizer = torch.optim.SGD(actor.parameters(), args.lr)
    else:
        optimizer = torch.optim.SGD(
            list(clf.parameters()) + list(actor.parameters()), args.lr)

    # watchlist
    losses, rewards, losses_zero, losses_pos, losses_grad, losses_bound = utils.get_average_meters(6)

    s_null = torch.from_numpy(np.array([[1, 0, 0]])).float()
    s_stan = torch.from_numpy(np.array([[-1, 0, 8]])).float()
    if args.use_cuda:
        s_null = s_null.cuda()
        s_stan = s_stan.cuda()

    if args.fix_init:
        env.reset()
        assert args.batch_size ==1
    s_list = []
    for ep_i in range(args.num_epochs):
        score = 0

        #u_list=[]

        if args.random_sampled:
            if args.sample_only_once==False or ep_i==0:
                N=args.batch_size*args.sim_len
                rth=np.random.random(N) * np.pi * 2
                rthdot=np.random.random(N) * 16 - 8

                # n = int(np.sqrt(N))
                # rth = np.linspace(-np.pi, np.pi, n)
                # rthdot = np.linspace(-8, 8, n)
                # rth, rthdot = np.meshgrid(rth, rthdot)
                # rth = rth.flatten()
                # rthdot = rthdot.flatten()

                s_list = np.stack((np.cos(rth), np.sin(rth), rthdot), axis=-1)

                # # TODO Debug only
                # dbg_len = 300
                # s = s_list
                # s = torch.from_numpy(s).float().cuda()
                # prev_s = None
                # for dbg_t in range(dbg_len):
                #     # plot first graph
                #     # plot next graph
                #     # plot vector field
                #     # plot multi graphs
                #     u = actor(s)
                #     s_next = dynamics(s, u, args)
                #     prev_s = s
                #     s = s_next
                #
                #     # plot previous
                #     prev_s = prev_s.detach().cpu().numpy()
                #     th = np.arctan2(prev_s[:, 1], prev_s[:, 0])
                #     thdot = prev_s[:, 2]
                #     plt.figure(figsize=(6, 6))
                #     plt.scatter(th, thdot)
                #     plt.axis('scaled')
                #     plt.xlim(-2*np.pi, 2*np.pi)
                #     plt.ylim(-8.0, 8.0)
                #     plt.savefig(ospj(viz_path,"dbg_%03d.png"%(dbg_t)))
                #     plt.close()
                #
                #     # plot vector field
                #     curr_s = s.detach().cpu().numpy()
                #     th1 = np.arctan2(curr_s[:, 1], curr_s[:, 0])
                #     thdot1 = curr_s[:, 2]
                #
                #     plt.figure(figsize=(6, 6))
                #     # plt.scatter(th, thdot)
                #     # plt.axis('scaled')
                #     # plt.xlim(-2 * np.pi, 2 * np.pi)
                #     # plt.ylim(-8.0, 8.0)
                #
                #     th_vec = th1 - th
                #     thdot_vec = thdot1 - thdot
                #     C = np.hypot(th_vec, thdot_vec)
                #     th_norm = th_vec / C
                #     thdot_norm = thdot_vec / C
                #
                #     Q = plt.quiver(th, thdot, th_norm, thdot_norm, C, units='xy', cmap=cm.gnuplot)
                #     plt.colorbar()
                #
                #     plt.savefig(ospj(viz_path, "vec_%03d.png" % (dbg_t)))
                #     plt.close()



        else:
            s_list = []
            u_list=[]
            # gather
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
                    if args.render: #TODO saved to files
                        env.render()

                    s_list.append(state)
                    u_list.append(u)
                    state = state_
        # train
        if args.random_sampled:
            # s = np.stack(s_list, axis=0)
            s = s_list
            s = torch.from_numpy(s).float()
            if args.use_cuda:
                s = s.cuda()
            u = actor(s)
        else:
            s = torch.cat(s_list, dim=0)  # (batch_size * sim_len, 3)
            u = torch.stack(u_list, dim=0)    # (batch_size * sim_len, 1)

        v = clf(s)                 # (batch_size * sim_len, 1)
        v_null = clf(s_null)       # (1, 1)

        v_stan = clf(s_stan)

        s_next = dynamics(s, u, args)
        v_next = clf(s_next)
        # print(s[0,:].detach().cpu().numpy(), s_next[0,:].detach().cpu().numpy(), #s[1,:].detach().cpu().numpy(),
        #       u[:,0].detach().cpu().numpy(),
        #       v[0].detach().cpu().numpy(), v[1].detach().cpu().numpy(), v_next[0].detach().cpu().numpy(), v_next[1].detach().cpu().numpy())
        # for dbg_i in range(args.batch_size):
            # if dbg_i in [3,4]:
            #     print("b=",dbg_i, "s=",s[dbg_i,:].detach().cpu().numpy(), s_next[dbg_i,:].detach().cpu().numpy(),v[dbg_i].detach().cpu().numpy(), v_next[dbg_i].detach().cpu().numpy() )
            #
            #     # dbg_s=s[dbg_i:dbg_i+1,:].detach()
            #     # dbg_u=actor(dbg_s)
            #     # dbg_s_next=dynamics(dbg_s, dbg_u, args)
            #     # dbg_v=clf(dbg_s)
            #     # dbg_v_next=clf(dbg_s_next)
            #     # dbg_loss=torch.mean(dbg_v_next-dbg_v)
            #     # dbg_grad = torch.autograd.grad(outputs=dbg_loss, inputs=dbg_u)
            #     # print(dbg_grad)
            #
            # else:
            #     print("b=", dbg_i, v[dbg_i].detach().cpu().numpy(), v_next[dbg_i].detach().cpu().numpy())
            # print("b=", dbg_i, v[dbg_i].detach().cpu().numpy(), v_next[dbg_i].detach().cpu().numpy())
        v_dot = (v_next - v) / args.dt

        # losses
        # if ep_i%1000==0:
        #     print(torch.mean(v).item(), torch.max(v).item(), torch.min(v).item())

        th_0 = torch.atan2(s[:,1],s[:,0])
        thd_0 = s[:, 2]
        v_out = v[torch.where(th_0**2+thd_0**2>1)]

        loss_zero = v_null**2 + torch.mean(nn.ReLU()(0.1-v_out)) #+ (0.1-torch.mean(v))**2 # v0=0
        loss_pos = torch.mean(torch.relu(args.margin_pos - v))  # v >= margin_pos
        loss_grad = torch.mean(torch.relu(args.margin_grad + v_dot)) # vdot<=-margin_grad
        loss_bound = torch.mean(torch.relu(v - args.margin_factor))  # v<=1

        loss_zero = args.weight_zero * loss_zero
        loss_pos = args.weight_pos * loss_pos
        loss_grad = args.weight_grad * loss_grad
        loss_bound = args.weight_bound * loss_bound

        loss = loss_zero + loss_pos + loss_grad + loss_bound

        rewards.update(score/args.batch_size)
        losses.update(loss.detach().cpu().item())
        losses_zero.update(loss_zero.detach().cpu().item())
        losses_pos.update(loss_pos.detach().cpu().item())
        losses_grad.update(loss_grad.detach().cpu().item())
        losses_bound.update(loss_bound.detach().cpu().item())

        writer.add_scalar("0_reward", score, ep_i)
        writer.add_scalar("1_loss", loss, ep_i)
        writer.add_scalar("2_loss_zero", loss_zero, ep_i)
        writer.add_scalar("3_loss_pos", loss_pos, ep_i)
        writer.add_scalar("4_loss_grad", loss_grad, ep_i)
        writer.add_scalar("5_loss_bound", loss_bound, ep_i)

        loss.backward()
        optimizer.step()

        # for dbg_i in range(args.batch_size):
        #     if dbg_i == 3:
        #
        #         # dbg_s = s[dbg_i:dbg_i + 1, :].detach()
        #         # dbg_u = actor(dbg_s)
        #         # dbg_s_next = dynamics(dbg_s, dbg_u, args)
        #         # dbg_v = clf(dbg_s)
        #         # dbg_v_next = clf(dbg_s_next)
        #         # dbg_loss = torch.mean(dbg_v_next - dbg_v)
        #         # dbg_grad = torch.autograd.grad(outputs=dbg_loss, inputs=dbg_u)
        #         # print(dbg_grad)
        #
        #         dbg_s = s.detach()
        #         dbg_u = actor(dbg_s)
        #         dbg_s_next = dynamics(dbg_s, dbg_u, args)
        #         dbg_v = clf(dbg_s)
        #         dbg_v_next = clf(dbg_s_next)
        #         dbg_v_dot = (dbg_v_next - dbg_v) / args.dt
        #         dbg_loss = torch.mean(torch.relu(args.margin_grad + dbg_v_dot))
        #         dbg_grad = torch.autograd.grad(outputs=dbg_loss, inputs=actor.linear_list[-1].weight)
        #         print("loss=", dbg_loss.detach().cpu().numpy(), "grad=",dbg_grad[0].detach().cpu().numpy().flatten())

        # print("u_max", torch.max(torch.abs(u)))

        # print & plot
        if ep_i % args.print_freq == 0:
            print("[%05d/%05d] reward:%.4f(%.4f) loss:%.4f(%.4f) zero:%.4f(%.4f) pos:%.4f(%.4f) grad:%.4f(%.4f) bound:%.4f(%.4f)"%(
                ep_i, args.num_epochs, rewards.val, rewards.avg, losses.val, losses.avg,
                losses_zero.val, losses_zero.avg, losses_pos.val, losses_pos.avg, losses_grad.val, losses_grad.avg,
                losses_bound.val, losses_bound.avg,
            ))

        if ep_i % args.viz_freq == 0:
            # # plot trace figure
            # th = torch.atan2(s[:, 1], s[:, 0])
            # plt.plot(th.detach().cpu().numpy(), s[:, 2].detach().cpu().numpy())
            # plt.axis('scaled')
            # plt.xlim(-np.pi, np.pi)
            # plt.ylim(-4, 4)
            # plt.savefig(ospj(viz_path, "1_e%05d_trace.png"%(ep_i)), bbox_inches='tight', pad_inches=0)
            # plt.close()

            # plot contour
            n_ths = 63
            n_thdots = 161
            ths = np.linspace(-np.pi, np.pi, n_ths)
            thdots = np.linspace(-8, 8, n_thdots)

            clf.eval()
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
            plt.savefig(ospj(viz_path, "2_e%05d_contour.png" % (ep_i)), bbox_inches='tight', pad_inches=0)
            plt.close()

            s = env.reset()
            for dbg_t in range(200):
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
    t1=time.time()
    print("Finished in %.4f seconds"%(t1-t0))


if __name__ == "__main__":
    main()

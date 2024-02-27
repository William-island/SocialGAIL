import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Batch

from .ppo import PPO
from gail_airl_ppo.network import GAILDiscrim, GraphDiscrim


class GAIL(PPO):

    def __init__(self, buffer_exp, graph_feature_channels, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=2048, mix_buffer=1,
                 batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=64, units_critic=64,
                 units_disc=64, epoch_ppo=40, epoch_disc=10,
                 clip_eps=0.2, lambd=0.97, coef_ent=0.0, max_grad_norm=10.0):     # coef_ent=0.0
        super().__init__(
            graph_feature_channels, state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        # self.disc = GAILDiscrim(
        #     state_shape=state_shape,
        #     action_shape=action_shape,
        #     hidden_units=units_disc,
        #     hidden_activation=nn.Tanh()
        # ).to(device)
        self.disc = GraphDiscrim(
            in_channels = graph_feature_channels,
            action_shape = action_shape,
            final_mlp_hidden_width = units_disc
        ).to(device)


        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc
        self.device = device

    def update(self, writer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            states, actions = self.buffer.sample(self.batch_size)[:2]
            states = Batch.from_data_list(states).to(self.device)
            # Samples from expert's demonstrations.
            states_exp, actions_exp = \
                self.buffer_exp.sample(self.batch_size)[:2]
            states_exp = Batch.from_data_list(states_exp).to(self.device)
            # Update discriminator.
            self.update_disc(states, actions, states_exp, actions_exp, writer)

        # We don't use reward signals here,
        states, actions, _, dones, log_pis, next_states = self.buffer.get()
        states = Batch.from_data_list(states).to(self.device)
        next_states = Batch.from_data_list(next_states).to(self.device)

        # Calculate rewards.
        rewards = self.disc.calculate_reward(states, actions)
        with torch.no_grad():
            pos_reward_rate =  (rewards > 0).float().mean().item()
        writer.add_scalar('rewards/pos_reward_rate', pos_reward_rate, self.learning_steps)

        # Update PPO using estimated rewards.
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_disc(self, states, actions, states_exp, actions_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, actions)
        logits_exp = self.disc(states_exp, actions_exp)

        ## Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        alpha = 0.0

        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = (1+alpha)*loss_exp + (1-alpha)*loss_pi
        # loss_pi = logits_pi.mean()
        # loss_exp = logits_exp.mean()
        # loss_disc = loss_pi - loss_exp


        self.optim_disc.zero_grad()
        loss_disc.backward()

        # # gradient_penalty
        # real_data = torch.cat([states_exp, actions_exp], dim=-1)
        # fake_data = torch.cat([states, actions], dim=-1)
        # gradient_penalty = self.calc_gradient_penalty(real_data, fake_data)
        # gradient_penalty.backward()

        self.optim_disc.step()



        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)



    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(self.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        
        interpolates = interpolates.to(self.device)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.disc.gp_forward(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10  # (LAMBDA)
        return gradient_penalty

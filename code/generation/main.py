import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from data_processing.mobility_data_processing import (
    generate_hourly_transactions,
    generate_slot_data,
    preprocess_sensor_data,
)
from data_processing.generate_external_data import download_meteo
from utils.models import Critic, Encoder, Generator
from utils.utils import add_conditions, grid_building

parser = argparse.ArgumentParser(allow_abbrev=False)

# ARGS FOR GENERATION 1st SCENARIO
parser.add_argument("--scenario", type=str, default="1st")
parser.add_argument("--cond_dim", type=int, default=2)
parser.add_argument("--input_dim", type=int, default=2)
parser.add_argument("--horizon", type=int, default=6 * 7)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--lr_rate", type=float, default=1e-4)
parser.add_argument("--num_epochs", type=int, default=250)
parser.add_argument("--train_percentage", type=float, default=0.8)
parser.add_argument("--grid_size", type=int, default=100)
parser.add_argument("--padding", type=int, default=1)
parser.add_argument("--kernel_size", type=int, default=3)
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--selected_zone", type=str, default="zone_2")

# # ARGS FOR GENERATION 2nd SCENARIO
# parser.add_argument('--scenario', type=str, default='2nd')
# parser.add_argument('--cond_dim', type=int, default=2)
# parser.add_argument('--input_dim', type=int, default=1)
# parser.add_argument('--horizon', type=int, default=6 * 7)
# parser.add_argument('--batch_size', type=int, default=5)
# parser.add_argument('--lr_rate', type=float, default=1e-4)
# parser.add_argument('--num_epochs', type=int, default=250)
# parser.add_argument('--train_percentage', type=float, default=0.8)
# parser.add_argument('--grid_size', type=int, default=100)
# parser.add_argument('--padding', type=int, default=1)
# parser.add_argument('--kernel_size', type=int, default=3)
# parser.add_argument('--hidden_dim', type=int, default=32)
# parser.add_argument('--latent_dim', type=int, default=100)
# parser.add_argument('--selected_zone', type=str, default='zone_2')
# parser.add_argument('--quantity', type=int, default=200)


# # ARGS FOR GENERATION 3rd SCENARIO
# parser.add_argument('--scenario', type=str, default='3rd')
# parser.add_argument('--cond_dim', type=int, default=3)
# parser.add_argument('--input_dim', type=int, default=2)
# parser.add_argument('--horizon', type=int, default=6 * 7)
# parser.add_argument('--batch_size', type=int, default=5)
# parser.add_argument('--lr_rate', type=float, default=1e-4)
# parser.add_argument('--num_epochs', type=int, default=250)
# parser.add_argument('--train_percentage', type=float, default=0.8)
# parser.add_argument('--grid_size', type=int, default=100)
# parser.add_argument('--padding', type=int, default=1)
# parser.add_argument('--kernel_size', type=int, default=3)
# parser.add_argument('--hidden_dim', type=int, default=64)
# parser.add_argument('--latent_dim', type=int, default=100)


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_model(model, model_save_path):
    torch.save(model.state_dict(), model_save_path)


def train_model(
    train_dataloader,
    val_dataloader,
    device,
    model_save_dir,
    s_coords,
    args,
):
    # Model parameters
    model_args = {
        "grid_size": args.grid_size,  # Size of the grid
        "input_dim": args.input_dim,  # Input dimension
        "cond_dim": args.cond_dim,  # Conditional dimension
        "hidden_dim": args.hidden_dim,  # Starting hidden dimension
        "latent_dim": args.latent_dim,  # Latent dimension
        "horizon": args.horizon,  # Sequence length
        "kernel_size": args.kernel_size,  # Kernel size
        "padding": args.padding,  # Padding
    }

    encoder = Encoder(model_args).to(device)
    generator = Generator(model_args).to(device)
    critic = Critic(model_args).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(generator.parameters()),
        lr=1e-4,
        betas=(0.5, 0.9),
    )
    opt_C = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

    # Training parameters
    vae_train_list = []
    vae_val_list = []
    g_train_list = []
    g_val_list = []
    c_train_list = []
    c_val_list = []
    one = torch.FloatTensor([1])[0]
    mone = one * -1
    mone = mone.to(device)
    one = one.to(device)
    for epoch in range(args.num_epochs):
        g_loss_batch = 0
        c_loss_batch = 0
        vae_loss_batch = 0
        critic.train()
        generator.train()
        encoder.train()

        for _, (data, conditions) in enumerate(train_dataloader):
            data = data.permute(0, 2, 1, 3, 4)
            data = data.to(device)
            conditions = conditions.permute(0, 2, 1, 3, 4)
            if args.scenario == "2nd":
                lat_indices = [lat for lat, _ in s_coords.values()]
                lon_indices = [lon for _, lon in s_coords.values()]
                mask_zero = conditions[:, 1, :, lat_indices, lon_indices] == 0
                conditions[:, 1, :, lat_indices, lon_indices] = torch.from_numpy(
                    np.where(
                        mask_zero,
                        0,
                        torch.log(
                            1 / (conditions[:, 1, :, lat_indices, lon_indices] + 1)
                        ),
                    )
                ).float()

            conditions = conditions.to(device)

            # Encoder conditions
            optimizer.zero_grad()
            z_cond, mu_cond, logvar_cond, indices, map = encoder(conditions)

            # Generator
            noise = torch.randn(data.shape[0], args.latent_dim).to(device)
            z_new = torch.cat([noise, z_cond], dim=1)
            fake_data, _ = generator(z_new, map, indices)

            # Critic
            opt_C.zero_grad()
            real_output = critic(data, conditions.detach())
            c_loss_real = torch.mean(real_output)
            c_loss_real.backward(mone, retain_graph=True)

            fake_output = critic(fake_data.detach(), conditions.detach())
            c_loss_fake = torch.mean(fake_output)
            c_loss_fake.backward(one)

            # Critic update
            c_loss = c_loss_fake - c_loss_real
            opt_C.step()

            for p in critic.parameters():
                p.data.clamp_(-0.01, 0.01)

            # Encoder and Generator update
            recon_loss = reconstruction_loss(fake_data, data)
            kull_loss = kl_loss(mu_cond, logvar_cond)
            loss_vae = recon_loss + 0.1 * kull_loss
            loss_vae.backward(one, retain_graph=True)
            fake_output = critic(fake_data, conditions)
            g_loss = -torch.mean(fake_output)
            g_loss.backward(one)
            optimizer.step()

            g_loss_batch += g_loss.item()
            c_loss_batch += c_loss.item()
            vae_loss_batch += loss_vae.item()

        g_loss = g_loss_batch / len(train_dataloader)
        c_loss = c_loss_batch / len(train_dataloader)
        loss = vae_loss_batch / len(train_dataloader)

        c_train_list.append(c_loss)
        vae_train_list.append(loss)
        g_train_list.append(g_loss)

        # Validation
        g_loss_val_batch = 0
        c_loss_val_batch = 0
        vae_loss_val_batch = 0
        encoder.eval()
        critic.eval()
        generator.eval()
        with torch.no_grad():
            for _, (data, conditions) in enumerate(val_dataloader):
                data = data.permute(0, 2, 1, 3, 4)
                data = data.to(device)
                conditions = conditions.permute(0, 2, 1, 3, 4)
                if args.scenario == "2nd":
                    lat_indices = [lat for lat, _ in s_coords.values()]
                    lon_indices = [lon for _, lon in s_coords.values()]
                    mask_zero = conditions[:, 1, :, lat_indices, lon_indices] == 0
                    conditions[:, 1, :, lat_indices, lon_indices] = torch.from_numpy(
                        np.where(
                            mask_zero,
                            0,
                            torch.log(
                                1 / (conditions[:, 1, :, lat_indices, lon_indices] + 1)
                            ),
                        )
                    ).float()

                conditions = conditions.to(device)

                # Encoder
                z_cond, mu_cond, logvar_cond, indices, map = encoder(conditions)

                # Generator
                noise = torch.randn(data.shape[0], args.latent_dim).to(device)
                z_new = torch.cat([noise, z_cond], dim=1)
                fake_data, _ = generator(z_new, map, indices)

                # Critic
                real_output = critic(data, conditions)
                c_loss_real = torch.mean(real_output)
                fake_output = critic(fake_data, conditions)
                c_loss_fake = torch.mean(fake_output)

                # Losses
                c_val_loss = c_loss_fake - c_loss_real

                recon_loss_val = reconstruction_loss(fake_data, data)
                kull_loss_val = kl_loss(mu_cond, logvar_cond)
                loss_val_vae = recon_loss_val + 0.1 * kull_loss_val

                g_val_loss = -torch.mean(fake_output)

                g_loss_val_batch += g_val_loss.item()
                c_loss_val_batch += c_val_loss.item()
                vae_loss_val_batch += loss_val_vae.item()

        g_loss_val = g_loss_val_batch / len(val_dataloader)
        c_loss_val = c_loss_val_batch / len(val_dataloader)
        loss_val = vae_loss_val_batch / len(val_dataloader)

        if len(vae_val_list) != 0:
            if loss_val < min(vae_val_list):
                generator_best = generator
                encoder_best = encoder

        g_train_list.append(g_loss)
        c_train_list.append(c_loss)
        vae_val_list.append(loss_val)
        print(
            f"Epoch {epoch} - Generator Loss: {g_loss} - VAE Loss: {loss} - Critic Loss : {c_loss} - Generator Loss Val: {g_loss_val} - VAE Loss Val: {loss_val} - Critic Loss Val: {c_loss_val}"
        )

    train_losses = [vae_train_list, g_train_list, c_train_list]
    val_losses = [vae_val_list, g_val_list, c_val_list]
    save_model(generator_best, model_save_dir + "/generator.pth")
    save_model(encoder_best, model_save_dir + "/encoder.pth")
    return train_losses, val_losses, encoder_best, generator_best


def reconstruction_loss(recon, data):
    loss_type = nn.MSELoss(reduction="sum")
    loss = loss_type(recon, data)

    return loss / (
        recon.shape[0]
        * recon.shape[1]
        * recon.shape[2]
        * recon.shape[3]
        * recon.shape[4]
    )


def kl_loss(z_mean, z_log_var):
    kl = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
    return kl.mean()


def generate(encoder_best, generator_best, conditions, device, args):
    encoder_best.eval()
    generator_best.eval()
    with torch.no_grad():
        conditions = conditions.permute(0, 2, 1, 3, 4)
        conditions = conditions.to(device)
        z_cond, _, _, indices, map = encoder_best(conditions)

        noise = torch.randn(conditions.shape[0], args.latent_dim).to(device)
        z_new = torch.cat([noise, z_cond], dim=1)
        fake_data, _ = generator_best(z_new, map, indices)

    return fake_data


if __name__ == "__main__":
    args, _ = parser.parse_known_args()

    data_dir = Path("../data")

    # Generate parkimeters data
    transaction_data = pd.read_csv(data_dir / "transaction_data.csv", index_col=0)

    hourly_transactions = generate_hourly_transactions(
        transaction_data, "transactions", freq="4h"
    )

    with open(data_dir / "anagraficaParcometro.json", "r") as f:
        anagrafica = json.load(f)

    final_parkimeters = {}
    for key in hourly_transactions.columns:
        final_parkimeters[key] = {}
        final_parkimeters[key]["lat"] = anagrafica[key]["lat"]
        final_parkimeters[key]["lng"] = anagrafica[key]["lng"]
        final_parkimeters[key]["data"] = hourly_transactions[key]

    # Generate slots data
    with open(data_dir / "AnagraficaStallo.json", "r") as f:
        slots = json.load(f)

    with open(data_dir / "KPlace_Signals.json", "r") as f:
        KPlace_signals = json.load(f)

    with open(data_dir / "StoricoStallo.json", "r") as f:
        storico_stallo = json.load(f)

    slots = pd.DataFrame(slots)

    storico_stallo = pd.DataFrame(storico_stallo)
    storico_stallo["start"] = pd.to_datetime(storico_stallo["start"])
    storico_stallo["end"] = pd.to_datetime(storico_stallo["end"])

    df_final = preprocess_sensor_data(KPlace_signals, slots, storico_stallo)
    occupied_slots = generate_slot_data(df_final, freq="4h")

    final_slots = {}
    for key in occupied_slots.columns:
        final_slots[key] = {}
        final_slots[key]["lat"] = slots[key]["lat"].values[0]
        final_slots[key]["lng"] = slots[key]["lng"].values[0]
        final_slots[key]["id_strada"] = slots[key]["id_strada"].values[0]
        final_slots[key]["data"] = occupied_slots[key]

    # Load mapping dictionary
    with open(data_dir / "mapping_dict.json", "r") as f:
        mapping_dict = json.load(f)

    # Common time range for all data
    time_start = max(
        min(final_slots[key]["data"].index[0] for key in final_slots.keys()),
        min(
            final_parkimeters[key]["data"].index[0] for key in final_parkimeters.keys()
        ),
    )
    time_end = min(
        max(final_slots[key]["data"].index[-1] for key in final_slots.keys()),
        max(
            final_parkimeters[key]["data"].index[-1] for key in final_parkimeters.keys()
        ),
    )

    # Grid matrix building
    if args.scenario == "1st" or args.scenario == "3rd":
        matrix, indices_slots, indices_parkimeter, scaler_slots, scaler_parkimeters = (
            grid_building(
                final_parkimeters,
                final_slots,
                args.grid_size,
                args.scenario,
                mapping_dict,
                time_start,
                time_end,
            )
        )
        tensor_matrix = torch.tensor(matrix, dtype=torch.float32).permute(3, 2, 0, 1)
        mask = (tensor_matrix != 0).float()
        if args.scenario == "3rd":
            # Generate weather data
            lat = 45.4642
            lon = 9.1900

            weather_data = download_meteo(time_start, time_end, lat, lon)

            weather_data.index = pd.to_datetime(weather_data.index)
            precipitation_data_4h = pd.DataFrame(
                weather_data["precipitation"].resample("4H").sum()
            )

            precipitation_data_4h["precipitation_mask"] = precipitation_data_4h[
                "precipitation"
            ].apply(lambda x: 1 if x > 0 else 0)

            mask_meteo = torch.zeros((mask.shape[0], 1, mask.shape[2], mask.shape[3]))

            for key in final_slots.keys():
                (lat_index, lon_index) = indices_slots[key]
                mask_meteo[:, 0, lat_index, lon_index] = torch.tensor(
                    precipitation_data_4h["precipitation_mask"].values
                )
            for key in final_parkimeters.keys():
                (lat_index, lon_index) = indices_parkimeter[key]
                mask_meteo[:, 0, lat_index, lon_index] = torch.tensor(
                    precipitation_data_4h["precipitation_mask"].values
                )

            mask = torch.cat([mask, mask_meteo], dim=1)

    elif args.scenario == "2nd":
        matrix, cond_matrix, indices_slots, scaler_slots = grid_building(
            final_parkimeters,
            final_slots,
            args.grid_size,
            args.scenario,
            mapping_dict,
            time_start,
            time_end,
        )
        tensor_matrix = torch.tensor(matrix, dtype=torch.float32).permute(3, 2, 0, 1)
        cond = torch.tensor(cond_matrix, dtype=torch.float32).permute(3, 2, 0, 1)
        mask = (tensor_matrix != 0).float()
        mask = torch.cat([mask, cond], dim=1)

    # Windowing
    X, mask_x = [], []
    for i in range(
        0, (len(tensor_matrix) // args.horizon) * args.horizon, args.horizon
    ):
        X.append(tensor_matrix[i : i + args.horizon])
        mask_x.append(mask[i : i + args.horizon])

    X = torch.stack(X)
    mask_x = torch.stack(mask_x)
    kwargs = {"pin_memory": True}

    num_samples = X.shape[0]
    train_size = int(args.train_percentage * num_samples)

    # Dataloader creation
    train_data, val_data, train_conditions, val_conditions = (
        X[:train_size],
        X[train_size:],
        mask_x[:train_size],
        mask_x[train_size:],
    )

    train_dataset = torch.utils.data.TensorDataset(train_data, train_conditions)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_conditions)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )

    # Training
    model_save_dir = "results"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    dir_name = "train"
    model_save_dir = os.path.join(model_save_dir, dir_name)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    print("Training generative model...")

    train_losses, val_losses, encoder_best, generator_best = train_model(
        train_dataloader,
        val_dataloader,
        device,
        model_save_dir,
        indices_slots,
        args,
    )

    # Generating
    print("Generating...")

    val_conditions_generation = add_conditions(
        val_conditions,
        indices_parkimeter,
        indices_slots,
        final_slots,
        final_parkimeters,
        mapping_dict,
        args,
    )

    fake_data = generate(
        encoder_best, generator_best, val_conditions_generation, device, args
    )

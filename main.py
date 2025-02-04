# %%
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.manifold import MDS
import os
import json
import numpy as np
import pandas as pd
import random
import holidays
import argparse
from pathlib import Path
from utils.models import Modelcomplete
from utils.utils import split, create_datasets, normalize_distance_matrix, haversine_matrix
from data_processing.generate_main_data import generate_hourly_transactions, generate_road_data, preprocess_sensor_data
from data_processing.generate_external_data import download_meteo, generate_events, generate_poi

parser = argparse.ArgumentParser(allow_abbrev=False)

# ARGS PER TRANSACTIONS
parser.add_argument('--data_type', type=str, default='transactions')
parser.add_argument('--target_channel', type=int, default=0)
parser.add_argument('--input_length', type=int, default=24 * 7 * 4)
parser.add_argument('--horizon', type=int, default=24 * 7)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--use_gps', type=bool, default=True)
parser.add_argument('--num_nodes', type=int, default=97)
parser.add_argument('--node_dim', type=int, default=16)
parser.add_argument('--input_dim', type=int, default=1)
parser.add_argument('--embed_dim', type=int, default=512)
parser.add_argument('--num_layer', type=int, default=1)
parser.add_argument('--temp_dim_tid', type=int, default=8)
parser.add_argument('--temp_dim_diw', type=int, default=8)
parser.add_argument('--time_of_day_size', type=int, default=24)
parser.add_argument('--day_of_week_size', type=int, default=7)
parser.add_argument('--if_T_i_D', type=bool, default=True)
parser.add_argument('--if_D_i_W', type=bool, default=True)
parser.add_argument('--if_node', type=bool, default=True)
parser.add_argument('--exogenous_dim', type=int, default=13)
parser.add_argument('--num_poi_types', type=int, default=7)
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--train_percentage', type=float, default=0.9)

# # ARGS PER AMOUNT
# parser.add_argument('--data_type', type=str, default='amount')
# parser.add_argument('--target_channel', type=int, default=0)
# parser.add_argument('--input_length', type=int, default=24 * 7 * 4)
# parser.add_argument('--horizon', type=int, default=24 * 7)
# parser.add_argument('--batch_size', type=int, default=128)
# parser.add_argument('--use_gps', type=bool, default=True)
# parser.add_argument('--num_nodes', type=int, default=97)
# parser.add_argument('--node_dim', type=int, default=16)
# parser.add_argument('--input_dim', type=int, default=1)
# parser.add_argument('--embed_dim', type=int, default=512)
# parser.add_argument('--num_layer', type=int, default=1)
# parser.add_argument('--temp_dim_tid', type=int, default=8)
# parser.add_argument('--temp_dim_diw', type=int, default=8)
# parser.add_argument('--time_of_day_size', type=int, default=24)
# parser.add_argument('--day_of_week_size', type=int, default=7)
# parser.add_argument('--if_T_i_D', type=bool, default=True)
# parser.add_argument('--if_D_i_W', type=bool, default=True)
# parser.add_argument('--if_node', type=bool, default=True)
# parser.add_argument('--exogenous_dim', type=int, default=13)
# parser.add_argument('--num_poi_types', type=int, default=7)
# parser.add_argument('--num_epochs', type=int, default=1000)
# parser.add_argument('--train_percentage', type=float, default=0.9)


# # ARGS PER ROADS
# parser.add_argument('--data_type', type=str, default='roads')
# parser.add_argument('--target_channel', type=int, default=0)
# parser.add_argument('--input_length', type=int, default=24 * 7 * 3)
# parser.add_argument('--horizon', type=int, default=24 * 7)
# parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--use_gps', type=bool, default=False)
# parser.add_argument('--num_nodes', type=int, default=56)
# parser.add_argument('--node_dim', type=int, default=16)
# parser.add_argument('--input_dim', type=int, default=1)
# parser.add_argument('--embed_dim', type=int, default=256)
# parser.add_argument('--num_layer', type=int, default=1)
# parser.add_argument('--temp_dim_tid', type=int, default=8)
# parser.add_argument('--temp_dim_diw', type=int, default=8)
# parser.add_argument('--time_of_day_size', type=int, default=24)
# parser.add_argument('--day_of_week_size', type=int, default=7)
# parser.add_argument('--if_T_i_D', type=bool, default=True)
# parser.add_argument('--if_D_i_W', type=bool, default=True)
# parser.add_argument('--if_node', type=bool, default=True)
# parser.add_argument('--exogenous_dim', type=int, default=13)
# parser.add_argument('--num_poi_types', type=int, default=7)
# parser.add_argument('--num_epochs', type=int, default=1000)
# parser.add_argument('--train_percentage', type=float, default=0.9)


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



# %%


def save_model(model, model_save_path):
    torch.save(model.state_dict(), model_save_path)


def train_model(
    train_dataloader,
    val_dataloader,
    criterion,
    device,
    model_save_dir,
    args,    
):
    # Model parameters
    model_args = {
        "num_nodes": args.num_nodes,  # Number of nodes
        "node_dim": args.node_dim,  # Spatial embedding dimension
        "input_len": args.input_length,  # Input sequence length
        "input_dim": args.input_dim,  # Input dimension 
        "embed_dim": args.embed_dim,  # Embedding dimension
        "output_len": args.horizon,  # Output sequence length
        "num_layer": args.num_layer,  # Number of MLP layers
        "temp_dim_tid": args.temp_dim_tid,  # Daily temporal embedding dimension
        "temp_dim_diw": args.temp_dim_diw,  # Weekly temporal embedding dimension
        "time_of_day_size": args.time_of_day_size,  # Number of hours in a day
        "day_of_week_size": args.day_of_week_size,  # Number of days in a week
        "if_T_i_D": args.if_T_i_D,  # Use daily temporal embedding
        "if_D_i_W": args.if_D_i_W,  # Use weekly temporal embedding
        "if_node": args.if_node,  # Use spatial embedding
        "if_gps": args.use_gps,  # Use GPS embedding
        "num_poi_types": args.num_poi_types,  # Number of POI types
        "exogenous_dim": args.exogenous_dim,  # Exogenous dimension
    }

    if args.use_gps:
        mds = MDS(n_components=model_args['node_dim'], dissimilarity='precomputed', random_state=42)
        gps_embeddings = mds.fit_transform(dist_matrix)
        gps_embeddings = torch.tensor(gps_embeddings, dtype=torch.float32).to(device)  # [N, node_dim]
        model_args["gps_embedding"] = gps_embeddings

    model = Modelcomplete(model_args).to(device)

    patience = 50
    early_stop = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    decay_rate = 0.5

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=decay_rate)

    best_epoch = 0
    train_losses = []
    val_losses = []
    min_lr = 1e-5
    best_validate_loss = np.inf
    validate_score_non_decrease_count = 0

    lr_update_interval = 5
    try:
        for epoch in range(args.num_epochs): 
            model.train()
            train_loss = 0.0
            for batch in train_dataloader:
                optimizer.zero_grad()

                seasonal = batch["seasonal"].to(device)
                trend = batch["trend"].to(device)
                residual = batch["residual"].to(device)

                exogenous = batch["exogenous"].to(device)
                    
                poi_tensor = batch["poi_data"].to(device)
                mask = batch["mask"].to(device)
                out, seasonal, residual, trend = model(seasonal, residual, trend, exogenous, poi_tensor, mask)

                targets = batch["targets"].to(device)
                targets = targets[..., [args.target_channel]]

                loss = criterion(out, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            val_pred = []
            val_true = []
            save_model(model, model_save_dir + "/model_last.pth")
            with torch.no_grad():
                for batch in val_dataloader:      
                    seasonal = batch["seasonal"].to(device)
                    trend = batch["trend"].to(device)
                    residual = batch["residual"].to(device)
                    
                    exogenous = batch["exogenous"].to(device)
                        
                    poi_tensor = batch["poi_data"].to(device)
                    mask = batch["mask"].to(device)
                    out, seasonal, residual, trend = model(seasonal, residual, trend, exogenous, poi_tensor, mask)

                    targets = batch["targets"].to(device)
                    targets = targets[..., [args.target_channel]]

                    loss = criterion(out, targets)

                    val_loss += loss.item()
                    val_pred.append(out.cpu().numpy())
                    val_true.append(targets.cpu().numpy())

            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)
            print("Epoch: ", epoch, "Train Loss: ", round(train_loss, 5), "Val Loss: ", round(val_loss,5))
            
            if scheduler:
                is_best_for_now = False
                if best_validate_loss > val_loss + 1e-5:
                    best_validate_loss = val_loss
                    is_best_for_now = True
                    validate_score_non_decrease_count = 0
                    model_best = model
                else:
                    validate_score_non_decrease_count += 1
                    
                if (validate_score_non_decrease_count+1) % lr_update_interval == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    if current_lr > min_lr:
                        print(f"Current learning rate: {current_lr}")
                        model.load_state_dict(model_best.state_dict())
                        scheduler.step()
                if is_best_for_now:
                    model_save_path_best = model_save_dir + "/model_best.pth"
                    save_model(model, model_save_path_best)
            # Early stop
            if early_stop and validate_score_non_decrease_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    except KeyboardInterrupt:
        print("Interrupted")
        save_model(model, model_save_dir + "/model_last.pth")

    return train_losses, val_losses, best_epoch, model_best



def predict(model, test_dataloader, args):
    
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():   
        for batch in test_dataloader:
            targets = batch["targets"].to(device)
            seasonal = batch["seasonal"].to(device)
            trend = batch["trend"].to(device)
            residual = batch["residual"].to(device)
            exogenous = batch["exogenous"].to(device)
                
            poi_tensor = batch["poi_data"].to(device)
            mask = batch["mask"].to(device)
            out, _, _, _ = model(seasonal, residual, trend, exogenous, poi_tensor, mask)

            out = out.squeeze(-1)
            targets = targets[..., args.target_channel]
            predictions.append(out.cpu().numpy())
            actuals.append(targets.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    return predictions, actuals

# %%

if __name__ == "__main__":

    args, _ = parser.parse_known_args()

    data_dir = Path("data")
    
    # Generate data
    if args.data_type == "transactions" or args.data_type == "amount":

        transaction_data = pd.read_csv(data_dir / "transaction_data.csv", index_col=0)

        hourly_transactions = generate_hourly_transactions(transaction_data, args.data_type)

        start_date = hourly_transactions.index.min()
        end_date = hourly_transactions.index.max()

        all_data_index = hourly_transactions.index

    elif args.data_type == "roads":
        
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

        # Generate roads data
        df_final = preprocess_sensor_data(KPlace_signals, slots, storico_stallo)

        if len(df_final) > 0:
            roads_data = generate_road_data(df_final, slots)

            start_date = roads_data.index.min()
            end_date = roads_data.index.max()
            all_data_index = roads_data.index


    # Generate weather data
    lat = 45.4642
    lon = 9.1900

    weather_data = download_meteo(start_date, end_date, lat, lon)

    weather_data = weather_data.loc[all_data_index]

    # Generate events data
    events = pd.read_csv(data_dir / "events.csv", index_col=0)

    south = 45.433
    west = 9.15
    north = 45.5
    east = 9.25

    events_data = generate_events(events, south, west, north, east)


    events_data.index = pd.to_datetime(events_data.index)
    events_data = events_data.reindex(
        pd.date_range(
            all_data_index.min(),
            all_data_index.max() + pd.Timedelta(days=1) - pd.Timedelta(hours=1),
            freq="H",
        )
    )
    events_data = events_data.fillna(method="ffill")

    events_data = events_data.loc[all_data_index]
    years = all_data_index.year.unique()

    it_holidays = pd.to_datetime(
        [date for year in years for date, _ in holidays.Italy(years=year).items()]
    )

    is_holiday = pd.DataFrame(
        index=all_data_index, columns=["is_holiday"], data=0
    )

    is_holiday.loc[is_holiday.index.normalize().isin(it_holidays), "is_holiday"] = 1

    easter = pd.date_range("2024-03-28", "2024-04-01", freq="D")

    days_christmas = ["12-23", "12-24", "12-25", "12-26", "12-27", "12-28", "12-29", "12-30", "12-31", "01-01", "01-02", "01-03", "01-04", "01-05", "01-06"]
    christmas = [pd.date_range(start=f"{year}-12-23", end=f"{year+1}-01-06", freq="D") for year in years]    

    christmas = pd.DatetimeIndex(np.concatenate(christmas))

    august = pd.date_range("2024-08-01", "2024-08-31", freq="D")

    days_our_holidays = easter.union(christmas).union(august)

    our_holidays = pd.DataFrame(index=all_data_index, columns=["our_holidays"], data=0)

    our_holidays.loc[our_holidays.index.normalize().isin(days_our_holidays), "our_holidays"] = 1

    exog_data = pd.concat([weather_data, events_data, is_holiday, our_holidays], axis=1)


    # Generate POI data
    if args.data_type == "transactions" or args.data_type == "amount":

        with open(data_dir / "anagraficaParcometro.json", "r") as f:
            anagrafica = json.load(f)
        
        poi_dist, poi_categories = generate_poi(anagrafica, south, west, north, east, args.data_type)


    elif args.data_type == "roads":
        
        with open(data_dir / "roads.json", "r") as f:
            anagrafica_roads = json.load(f)

        poi_dist, poi_categories = generate_poi(anagrafica_roads, south, west, north, east, args.data_type)

    poi_categories = np.expand_dims(poi_categories.values, axis=-1).astype(np.float32)
    poi_dist = np.expand_dims(poi_dist.values, axis=-1).astype(np.float32)

    # Mask to consider only POIs within 0.5 km from the parkimeter
    mask = poi_dist <= 0.5

    poi_dist_masked = poi_dist * mask

    # Normalize distance matrix 
    denominator = poi_dist_masked.max() - poi_dist_masked.min()
    if denominator == 0:
        pass
    else:
        poi_dist_masked = (poi_dist_masked - poi_dist_masked.min()) / denominator

    poi_data_ = np.concatenate([poi_categories, poi_dist_masked], axis=-1)

    poi_tensor = torch.tensor(
        poi_data_, dtype=torch.float32
    )

    mask = torch.tensor(mask, dtype=torch.float32)

    if args.use_gps:

        parcometri = anagrafica.copy()

        parcometri = pd.DataFrame(parcometri)
        parcometri = parcometri[["id", "lat", "lng"]]
        parcometri["id"] = parcometri["id"].astype(float)

        parcometri = parcometri.T.to_dict()

        gps_coordinates = torch.tensor(
            [
                [float(parcometri[parcometro]["lat"]), float(parcometri[parcometro]["lng"])]
                for parcometro in parcometri.keys()
            ]
        )

        dist_matrix = haversine_matrix(gps_coordinates)
        dist_matrix = normalize_distance_matrix(dist_matrix)

    if args.data_type == "transactions" or args.data_type == "amount":
        data_scaler, train_data_with_features, val_data_with_features, test_data_with_features = split(hourly_transactions, exog_data, args.train_percentage) 
    elif args.data_type == "roads":
        data_scaler, train_data_with_features, val_data_with_features, test_data_with_features = split(roads_data, exog_data, args.train_percentage)

    train_dataset, val_dataset, test_dataset = create_datasets(train_data_with_features, val_data_with_features, test_data_with_features, poi_tensor, mask, args)

    # Dataloader creation
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Training
    model_save_dir = "results"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    dir_name = "train"
    model_save_dir = os.path.join(model_save_dir, dir_name)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    criterion = nn.HuberLoss(reduction="mean")

    print("Training model...")
        
    train_losses, val_losses, best_epoch, model_best = train_model(
        train_dataloader,
        val_dataloader,
        criterion,  
        device,
        model_save_dir,
        args,
    )

    # Prediction
    print("Predicting...")
    pred_series, actual_series = predict(model_best, test_dataloader, args)


    pred_series = np.maximum(pred_series, 0)
    actual_series1 = data_scaler.inverse_transform(
        actual_series.reshape(-1, actual_series.shape[2])
    ).reshape(actual_series.shape)
    pred_series1 = data_scaler.inverse_transform(
        pred_series.reshape(-1, pred_series.shape[2])
    ).reshape(pred_series.shape)



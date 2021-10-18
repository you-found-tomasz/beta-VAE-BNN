from ray import tune
import csv


def load_config(network_enc, network_dec, raytune, config_file_name):

    config_enc = dict()
    config_dec = dict()

    if raytune == "True":
        full_config = {
                "lr": tune.loguniform(1e-6, 1e-1),
                "batch_size": tune.choice([4, 8, 16, 32, 64, 128]),
                "latent_space": tune.choice([16])
                       }

        # encoder configs
        if network_enc == "deep":
            config_enc = {
                "enc_number_of_layers": tune.choice([2, 3, 4, 5]),
                "enc_layer_1_size": tune.choice([4, 8, 16, 128, 512, 1024, 2048]),
                "enc_layer_2_size": tune.choice([4, 8, 16, 128, 512, 1024, 2048]),
                "enc_layer_3_size": tune.choice([4, 8, 16, 128, 512, 1024, 2048]),
                "enc_layer_4_size": tune.choice([4, 8, 16, 128, 512, 1024, 2048]),
                "enc_layer_5_size": tune.choice([4]),
                "enc_layer_6_size": tune.choice([4]),
                "enc_layer_7_size": tune.choice([4]),
                "enc_layer_1_bias": tune.choice([True, False]),
                "enc_layer_2_bias": tune.choice([True, False]),
                "enc_layer_3_bias": tune.choice([True, False]),
                "enc_layer_4_bias": tune.choice([True, False]),
                "enc_layer_5_bias": tune.choice([True]),
                "enc_layer_6_bias": tune.choice([True]),
                "enc_drop_0": tune.choice([0, 0.05, 0.1]),
                "enc_drop_1": tune.choice([0, 0.05, 0.1]),
                "enc_drop_2": tune.choice([0, 0.05, 0.1]),
                "enc_drop_3": tune.choice([0, 0.05, 0.1]),
                "enc_drop_4": tune.choice([0, 0.05, 0.1]),
                "enc_activation": tune.choice(["relu", "tanh"])
            }

        elif network_enc == "Resnet":
            config_enc = {
            "first_conv": tune.choice([False]),
            "maxpool1": tune.choice([False]),
            }
        else:
            pass

        # decoder configs
        if network_dec == "deep":
            config_dec = {
                "dec_number_of_layers": tune.choice([2, 3, 4]),
                "dec_layer_1_size": tune.choice([4, 8, 16, 128, 512, 1024]),
                "dec_layer_2_size": tune.choice([4, 8, 16, 128, 512, 1024]),
                "dec_layer_3_size": tune.choice([4, 8, 16, 128, 512, 1024]),
                "dec_layer_4_size": tune.choice([4, 8, 16, 128, 512, 1024]),
                "dec_layer_5_size": tune.choice([4]),
                "dec_layer_6_size": tune.choice([4]),
                "dec_layer_7_size": tune.choice([4]),
                "dec_layer_1_bias": tune.choice([True, False]),
                "dec_layer_2_bias": tune.choice([True, False]),
                "dec_layer_3_bias": tune.choice([True, False]),
                "dec_layer_4_bias": tune.choice([True, False]),
                "dec_layer_5_bias": tune.choice([True]),
                "dec_layer_6_bias": tune.choice([True]),
                "dec_drop_0": tune.choice([0, 0.05, 0.1]),
                "dec_drop_1": tune.choice([0, 0.05, 0.1]),
                "dec_drop_2": tune.choice([0, 0.05, 0.1]),
                "dec_drop_3": tune.choice([0, 0.05, 0.1]),
                "dec_drop_4": tune.choice([0, 0.05, 0.1]),
                "dec_activation": tune.choice(["relu", "tanh"])
            }

        elif network_dec == "Resnet":
            pass

        else:
            pass

        full_config.update(config_enc)
        full_config.update(config_dec)

    else:

        filename = config_file_name

        float_list = ["lr", "enc_drop_0", "enc_drop_1", "enc_drop_2", "enc_drop_3", "enc_drop_4", "dec_drop_0",
                      "dec_drop_1", "dec_drop_2", "dec_drop_3", "dec_drop_4"]
        bool_list = ["first_conv", "maxpool1", "enc_layer_1_bias", "enc_layer_2_bias", "enc_layer_3_bias",
                     "enc_layer_4_bias", "enc_layer_5_bias", "enc_layer_6_bias", "dec_layer_1_bias", "dec_layer_2_bias",
                     "dec_layer_3_bias", "dec_layer_4_bias", "dec_layer_5_bias", "dec_layer_6_bias"]
        str_list = ["enc_activation", "dec_activation"]

        config_file = dict()
        with open(filename, mode='r') as inp:
            reader = csv.reader(inp)
            for rows in reader:
                if rows[0] in float_list:
                    config_file.update({rows[0]: float(rows[1].replace(',', '.'))})
                elif rows[0] in bool_list:
                    config_file.update({rows[0]: bool(rows[1])})
                elif rows[0] in str_list:
                    config_file.update({rows[0]: str(rows[1])})
                else:
                    config_file.update({rows[0]: int(rows[1])})

        full_config = config_file

    return full_config

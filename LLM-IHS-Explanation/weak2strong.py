from load_data import get_data, load_conv
from load_model import get_model, step_forward
import numpy as np
import os
from w2s_utils import get_layer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import accelerate
from visualization import topk_intermediate_confidence_heatmap, accuracy_line


norm_prompt_path = './exp_data/normal_prompt.csv'
jailbreak_prompt_path = './exp_data/jailbreak_prompt.csv'
malicious_prompt_path = './exp_data/malicious_prompt.csv'


def load_exp_data(shuffle_seed=None, use_conv=False, model_name=None):
    normal_inputs = get_data(norm_prompt_path, shuffle_seed)
    malicious_inputs = get_data(malicious_prompt_path, shuffle_seed)
    if os.path.exists(jailbreak_prompt_path):
        jailbreak_inputs = get_data(jailbreak_prompt_path, shuffle_seed)
    else:
        jailbreak_inputs = None
    if use_conv and model_name is None:
        raise ValueError("please set model name for load")
    if use_conv:
        normal_inputs = [load_conv(model_name, _) for _ in normal_inputs]
        malicious_inputs = [load_conv(model_name, _) for _ in malicious_inputs]
        jailbreak_inputs = [load_conv(model_name, _) for _ in jailbreak_inputs] if jailbreak_inputs is not None else None
    return normal_inputs, malicious_inputs, jailbreak_inputs


class Weak2StrongClassifier:
    def __init__(self, return_report=True, return_visual=False):
        self.return_report = return_report
        self.return_visual = return_visual

    @staticmethod
    def _process_data(forward_info):
        features = []
        labels = []
        for key, value in forward_info.items():
            for hidden_state in value["hidden_states"]:
                features.append(hidden_state.flatten())
                labels.append(value["label"])

        features = np.array(features)
        labels = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def svm(self, forward_info):
        X_train, X_test, y_train, y_test = self._process_data(forward_info)
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)
        report = None
        if self.return_report:
            print("SVM Test Classification Report:")
            print(classification_report(y_test, y_pred, zero_division=0.0))
        if self.return_visual:
            report = classification_report(y_test, y_pred, zero_division=0.0, output_dict=True)
        return X_test, y_pred, report

    def mlp(self, forward_info):
        X_train, X_test, y_train, y_test = self._process_data(forward_info)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.01,
                            solver='adam', verbose=0, random_state=42,
                            learning_rate_init=.01)

        mlp.fit(X_train_scaled, y_train)
        y_pred = mlp.predict(X_test_scaled)
        report = None
        if self.return_report:
            print("MLP Test Classification Report:")
            print(classification_report(y_test, y_pred, zero_division=0.0))
        if self.return_visual:
            report = classification_report(y_test, y_pred, zero_division=0.0, output_dict=True)
        return X_test, y_pred, report


class Weak2StrongExplanation:
    def __init__(self, model_path, layer_nums=32, return_report=True, return_visual=True):
        self.model, self.tokenizer = get_model(model_path)
        self.model_name = model_path.split("/")[-1]
        self.layer_sums = layer_nums + 1
        self.forward_info = {}
        self.return_report = return_report
        self.return_visual = return_visual

    def get_forward_info(self, inputs_dataset, class_label, debug=True):
        offset = len(self.forward_info)
        for _, i in enumerate(inputs_dataset):
            if debug and _ > 100:
                break
            list_hs, tl_pair = step_forward(self.model, self.tokenizer, i)
            last_hs = [hs[:, -1, :] for hs in list_hs]
            self.forward_info[_ + offset] = {"hidden_states": last_hs, "top-value_pair": tl_pair, "label": class_label}

    def explain(self, datasets, classifier_list=None, debug=True, accuracy=True):
        self.forward_info = {}
        if classifier_list is None:
            classifier_list = ["svm", "mlp"]
        forward_info = {}
        if isinstance(datasets, list):
            for class_num, dataset in enumerate(datasets):
                self.get_forward_info(dataset, class_num, debug=debug)
        elif isinstance(datasets, dict):
            for class_key, dataset in datasets.items():
                self.get_forward_info(dataset, class_key, debug=debug)
        
        classifier = Weak2StrongClassifier(self.return_report, self.return_visual)

        rep_dict = {}
        if "svm" in classifier_list:
            rep_dict["svm"] = {}
            for _ in range(0, self.layer_sums):
                x, y, rep = classifier.svm(get_layer(self.forward_info, _))
                rep_dict["svm"][_] = rep

        if "mlp" in classifier_list:
            rep_dict["mlp"] = {}
            for _ in range(0, self.layer_sums):
                x, y, rep = classifier.mlp(get_layer(self.forward_info, _))
                rep_dict["mlp"][_] = rep
        
        if not self.return_visual:
            return
        
        if accuracy and classifier_list != []:
            accuracy_line(rep_dict, self.model_name)

    def vis_heatmap(self, dataset, left=0, right=33, debug=True, model_name=""):
        self.forward_info = {}
        self.get_forward_info(dataset, 0, debug=debug)
        topk_intermediate_confidence_heatmap(self.forward_info, left=left, right=right,model_name=model_name)



# class Weak2StrongExplanation_VLM:
#     def __init__(self, model, tokenizer, layer_nums=32, return_report=True, return_visual=True):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.layer_sums = layer_nums + 1
#         self.forward_info = {}
#         self.return_report = return_report
#         self.return_visual = return_visual

#     def get_forward_info(self, inputs_dataset, class_label, debug=True):
#         offset = len(self.forward_info)
#         for _, i in enumerate(inputs_dataset):
#             if debug and _ > 100:
#                 break
#             list_hs, tl_pair = step_forward_vlm(self.model, self.tokenizer, i)
#             last_hs = [hs[:, -1, :] for hs in list_hs]
#             self.forward_info[_ + offset] = {"hidden_states": last_hs, "top-value_pair": tl_pair, "label": class_label}

#     def explain(self, datasets, classifier_list=None, debug=True, accuracy=True):
#         self.forward_info = {}
#         if classifier_list is None:
#             classifier_list = ["svm", "mlp"]
#         forward_info = {}
#         if isinstance(datasets, list):
#             for class_num, dataset in enumerate(datasets):
#                 self.get_forward_info(dataset, class_num, debug=debug)
#         elif isinstance(datasets, dict):
#             for class_key, dataset in datasets.items():
#                 self.get_forward_info(dataset, class_key, debug=debug)
        
#         classifier = Weak2StrongClassifier(self.return_report, self.return_visual)

#         rep_dict = {}
#         if "svm" in classifier_list:
#             rep_dict["svm"] = {}
#             for _ in range(0, self.layer_sums):
#                 x, y, rep = classifier.svm(get_layer(self.forward_info, _))
#                 rep_dict["svm"][_] = rep

#         if "mlp" in classifier_list:
#             rep_dict["mlp"] = {}
#             for _ in range(0, self.layer_sums):
#                 x, y, rep = classifier.mlp(get_layer(self.forward_info, _))
#                 rep_dict["mlp"][_] = rep
        
#         if not self.return_visual:
#             return
        
#         if accuracy and classifier_list != []:
#             accuracy_line(rep_dict, self.model_name)

#     def vis_heatmap(self, dataset, left=0, right=33, debug=True, model_name=""):
#         self.forward_info = {}
#         self.get_forward_info(dataset, 0, debug=debug)
#         topk_intermediate_confidence_heatmap(self.forward_info, left=left, right=right,model_name=model_name)
            
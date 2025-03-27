from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import re
import numpy as np

np.random.seed(80)

df_X = pd.read_csv("train_data.csv")
df_Y = pd.read_csv("train_label.csv")
df = pd.concat([df_X, df_Y], axis=1)
df = df.dropna()


def extract_pin(address):
    # Using regular expression to find the last "word" (sequence of characters without spaces)
    pin_code = re.findall(r'\b\S+$', address)
    if pin_code:
        return pin_code[0]
    else:
        return None

# Add 'PIN CODE' column to DataFrame
df['PIN CODE'] = df['MAIN_ADDRESS'].apply(extract_pin)

# Add 2 columns with ratios - feature engineering
df.insert(df.columns.get_loc('BEDS'), 'SQFT_BATH_RATIO', df['PROPERTYSQFT'] / df['BATH'].replace(0, 1))
df.insert(df.columns.get_loc('BEDS'), 'PRICE_SQFT_RATIO', df['PRICE'] / df['PROPERTYSQFT'])

beds_index = df.columns.get_loc('BEDS')
df.insert(beds_index, 'PIN CODE', df.pop('PIN CODE'))

test = pd.read_csv("test_data.csv")
test = test.dropna()

test['PIN CODE'] = test['MAIN_ADDRESS'].apply(extract_pin)


test.insert(loc=test.shape[1], column='SQFT_BATH_RATIO', value=test['PROPERTYSQFT'] / test['BATH'].replace(0, 1))
test.insert(loc=test.shape[1], column='PRICE_SQFT_RATIO', value=test['PRICE'] / test['PROPERTYSQFT'])

pin_code_column = test.pop('PIN CODE')
test.insert(loc=test.shape[1], column='PIN CODE', value=pin_code_column)

# Columns to remove
columns_to_remove = ['BROKERTITLE', 'ADDRESS', 'FORMATTED_ADDRESS', 'LONG_NAME', 'STREET_NAME', 'ADMINISTRATIVE_AREA_LEVEL_2', 'STATE', 'MAIN_ADDRESS']

# Remove columns
df.drop(columns=columns_to_remove, inplace=True)

test.drop(columns=columns_to_remove, inplace=True)

df.drop('LATITUDE', axis=1, inplace=True)
df.drop('LONGITUDE', axis=1, inplace=True)

categorical_feats = ["LOCALITY", "SUBLOCALITY", "TYPE", "PIN CODE"]

def check_mean(df, feature):
  res = dict({})
  for value in pd.unique(df[feature]):
    filt = df[feature] == value
    res[value] = df.loc[filt, "BEDS"].mean()

  return res

for feat in categorical_feats:
  ordered = check_mean(df, feat)
  df[feat] = df[feat].map(ordered)
  test[feat] = test[feat].map(ordered)

df_normalized = df.copy()

for column in df.drop(columns=["BEDS"]).columns:
     df_normalized[column] = df[column].map(lambda i: np.log(i*10) if i > 0 else 0)

df_normalized = df_normalized.drop(columns=["BEDS"])
df_normalized["BEDS"] = df["BEDS"]

test_normalized = test.copy()

for column in test.columns:
     test_normalized[column] = test_normalized[column].map(lambda i: np.log(i*10) if i > 0 else 0)

class sequential(object):
    def __init__(self, *args):

        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        return self.params[name]

    def get_grads(self, name):
        return self.grads[name]

    def gather_params(self):

        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):

        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def load(self, pretrained):

        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print(
                        "Loading Params: {} Shape: {}".format(n, layer.params[n].shape)
                    )


class fc(object):
    def __init__(self, input_dim, output_dim, init_scale=0.002, name="fc"):

        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(input_dim, output_dim)
        self.params[self.b_name] = np.zeros(output_dim)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None

    def forward(self, feat):
        output = None
        assert (
            len(feat.shape) == 2 and feat.shape[-1] == self.input_dim
        ), "But got {} and {}".format(feat.shape, self.input_dim)

        output = np.dot(feat, self.params[self.w_name]) + self.params[self.b_name]
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        dfeat, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        assert (
            len(feat.shape) == 2 and feat.shape[-1] == self.input_dim
        ), "But got {} and {}".format(feat.shape, self.input_dim)
        assert (
            len(dprev.shape) == 2 and dprev.shape[-1] == self.output_dim
        ), "But got {} and {}".format(dprev.shape, self.output_dim)

        dW = np.dot(feat.T, dprev)
        db = np.sum(dprev, axis=0)
        dfeat = np.dot(dprev, self.params[self.w_name].T)

        self.grads[self.w_name] = dW
        self.grads[self.b_name] = db
        self.meta = None
        return dfeat


class leaky_relu(object):
    def __init__(self, negative_slope=0.01, name="leaky_relu"):

        self.negative_slope = negative_slope
        self.name = name
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, feat):
        output = None
        output = np.where(feat > 0, feat, feat * self.negative_slope)

        self.meta = feat
        return output

    def backward(self, dprev):

        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        dfeat = None
        dfeat = dprev * np.where(feat > 0, 1, self.negative_slope)

        self.meta = None
        return dfeat


class dropout(object):
    def __init__(self, keep_prob, seed=None, name="dropout"):

        self.name = name
        self.params = {}
        self.grads = {}
        self.keep_prob = keep_prob
        self.meta = None
        self.kept = None
        self.is_training = False
        self.rng = np.random.RandomState(seed)
        assert (
            keep_prob >= 0 and keep_prob <= 1
        ), "Keep Prob = {} is not within [0, 1]".format(keep_prob)

    def forward(self, feat, is_training=True, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        kept = None
        output = None

        if is_training and self.keep_prob:
            kept = self.rng.rand(*feat.shape)
            kept = np.where(kept < self.keep_prob, 1, 0)/self.keep_prob
        else:
            kept = np.ones_like(feat)

        output = feat * kept

        self.kept = kept
        self.is_training = is_training
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        dfeat = None
        if feat is None:
            raise ValueError("No forward function called before for this module!")

        if self.is_training and self.keep_prob:
            dfeat = dprev * self.kept / self.keep_prob
        else:
            dfeat = dprev

        self.is_training = False
        self.meta = None
        return dfeat

class MeanSquaredError(object):
    def __init__(self, size_average=True):

        self.size_average = size_average
        self.prediction = None
        self.target = None

    def forward(self, prediction, target):
        loss = None
        n = prediction.shape[0]
        squared_error = np.square(prediction - target)
        if self.size_average:
            loss = np.mean(squared_error)
        else:
            loss = np.sum(squared_error)

        self.prediction = prediction
        self.target = target
        return loss

    def backward(self):
        prediction = self.prediction
        target = self.target
        if prediction is None:
            raise ValueError("No forward function called before for this module!")
        grad = None
        batch_size = target.shape[0]
        grad = 2 * (prediction - target)
        if self.size_average:
            grad /= batch_size
        self.prediction = None
        self.target = None
        return grad

class cross_entropy(object):
    def __init__(self, size_average=True):

        self.size_average = size_average
        self.logit = None
        self.label = None

    def forward(self, feat, label):
        logit = softmax(feat)
        loss = None

        n = feat.shape[0]
        log_likelihood = -np.log(logit[np.arange(n), label])
        if self.size_average:
            loss = np.mean(log_likelihood)
        else:
            loss = np.sum(log_likelihood)
        self.logit = logit
        self.label = label
        return loss

    def backward(self):
        logit = self.logit
        label = self.label
        if logit is None:
            raise ValueError("No forward function called before for this module!")
        dlogit = None
        batch_size = label.shape[0]
        dlogit = logit.copy()
        dlogit[np.arange(batch_size), label] -= 1
        if self.size_average:
            dlogit /= batch_size
        self.logit = None
        self.label = None
        return dlogit

def softmax(feat):
    scores = None
    feat = feat - np.max(feat, axis=-1, keepdims=True)
    exp_scores = np.exp(feat)
    scores = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    return scores

class Module(object):
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, feat, is_training=True, seed=None):
        output = feat
        for layer in self.net.layers:
            if isinstance(layer, dropout):
                output = layer.forward(output, is_training, seed)
            else:
                output = layer.forward(output)
        self.net.gather_params()
        return output

    def backward(self, dprev):
        for layer in self.net.layers[::-1]:
            dprev = layer.backward(dprev)
        self.net.gather_grads()
        return dprev

class FullyConnectedNetwork(Module):
   def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
       self.net = sequential(
           fc(len(df_normalized.columns)-1, 144, 5e-2, name="fc1"),
           leaky_relu(name="relu1"),
           fc(144, 256, 5e-2, name="fc2"),
           leaky_relu(name="relu2"),
           fc(256, 272, 5e-2, name="fc3"),
           leaky_relu(name="relu3"),
           fc(272, df_normalized["BEDS"].max()+1, 5e-2, name="fc4"),
       )


class DropoutNetTest(Module):
   def __init__(self, keep_prob=0, dtype=np.float32, seed=None):


       self.dropout = dropout
       self.seed = seed
       self.net = sequential(
           fc(3072, 500, 1e-2, name="fc1"),
           dropout(keep_prob, seed=seed),
           leaky_relu(name="relu1"),
           fc(500, 500, 1e-2, name="fc2"),
           leaky_relu(name="relu2"),
           fc(500, 10, 1e-2, name="fc3"),
       )

class Optimizer(object):

    def __init__(self, net, lr=1e-4):
        self.net = net 
        self.lr = lr

    def step(self):
        if hasattr(self.net, "preprocess") and self.net.preprocess is not None:
            self.update(self.net.preprocess)
        if hasattr(self.net, "rnn") and self.net.rnn is not None:
            self.update(self.net.rnn)
        if hasattr(self.net, "postprocess") and self.net.postprocess is not None:
            self.update(self.net.postprocess)

        if (
            not hasattr(self.net, "preprocess")
            and not hasattr(self.net, "rnn")
            and not hasattr(self.net, "postprocess")
        ):
            for layer in self.net.layers:
                self.update(layer)

class SGD(Optimizer):

    def __init__(self, net, lr=1e-4):
        self.net = net
        self.lr = lr

    def update(self, layer):
        for n, dv in layer.grads.items():
            layer.params[n] -= self.lr * dv


class Adam(Optimizer):

    def __init__(self, net, lr=1e-3, beta1=0.95, beta2=0.999, t=0, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t

    def update(self, layer):
        self.t += 1
        for n, dv in layer.grads.items():
            if n not in self.mt:
                self.mt[n] = np.zeros_like(dv)
                self.vt[n] = np.zeros_like(dv)
            self.mt[n] = self.beta1 * self.mt[n] + (1 - self.beta1) * dv
            self.vt[n] = self.beta2 * self.vt[n] + (1 - self.beta2) * (dv ** 2)

            mt_ = self.mt[n] / (1 - self.beta1 ** self.t)
            vt_ = self.vt[n] / (1 - self.beta2 ** self.t)

            layer.params[n] -= self.lr * mt_ / (np.sqrt(vt_) + self.eps)

class DataLoader(object):

    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.indices = np.asarray(range(data.shape[0]))

    def _reset(self):
        self.indices = np.asarray(range(self.data.shape[0]))

    def _shuffle(self):
        np.random.shuffle(self.indices)

    def get_batch(self):
        if len(self.indices) < self.batch_size:
            self._reset()
            self._shuffle()
        indices_curr = self.indices[0 : self.batch_size]
        data_batch = self.data[indices_curr]
        labels_batch = self.labels[indices_curr]
        self.indices = np.delete(self.indices, range(self.batch_size))
        return data_batch, labels_batch


def compute_acc(model, data, labels, num_samples=None, batch_size=100):

    # N = data.shape[0]
    # if num_samples is not None and N > num_samples:
    #     indices = np.random.choice(N, num_samples)
    #     N = num_samples
    #     data = data[indices]
    #     labels = labels[indices]

    # num_batches = N // batch_size
    # if N % batch_size != 0:
    #     num_batches += 1
    # total_acc = 0
    # for i in range(num_batches):
    #     start = i * batch_size
    #     end = (i + 1) * batch_size
    #     output = model.forward(data[start:end], False)
    #     # print(output[0])
    #     acc = np.sum((output == labels[start:end]))
    #     total_acc += acc
    # acc = total_acc/batch_size
    # return acc

    N = data.shape[0]
    if num_samples is not None and N > num_samples:
        indices = np.random.choice(N, num_samples)
        N = num_samples
        data = data[indices]
        labels = labels[indices]

    num_batches = N // batch_size
    if N % batch_size != 0:
        num_batches += 1
    preds = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        output = model.forward(data[start:end], False)
        scores = softmax(output)
        pred = np.argmax(scores, axis=1)
        preds.append(pred)
    preds = np.hstack(preds)
    accuracy = np.mean(preds == labels)
    return accuracy

def train_net(
    data,
    model,
    loss_func,
    optimizer,
    batch_size,
    max_epochs,
    lr_decay=1.0,
    lr_decay_every=1000,
    show_every=10,
    verbose=False,
):

    data_train, labels_train = small_data_dict["data_train"]
    data_val, labels_val = small_data_dict["data_val"]
    dataloader = DataLoader(data_train, labels_train, batch_size)
    opt_val_acc = 0.0
    opt_params = None
    loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    iters_per_epoch = int(max(data_train.shape[0] / batch_size, 1))
    max_iters = int(iters_per_epoch * max_epochs)

    for epoch in range(max_epochs):

        iter_start = epoch * iters_per_epoch
        iter_end = (epoch + 1) * iters_per_epoch

        if epoch % lr_decay_every == 0 and epoch > 0:
            optimizer.lr = optimizer.lr * lr_decay
            print("Decaying learning rate of the optimizer to {}".format(optimizer.lr))


        for iter in range(iter_start, iter_end):
            data_batch, labels_batch = dataloader.get_batch()

            pred = model.forward(data_batch)
            loss = loss_func.forward(pred, labels_batch)
            model.backward(loss_func.backward())
            optimizer.step()
            loss_hist.append(loss)

            if verbose and iter % show_every == 0:
                print(
                    "(Iteration {} / {}) loss: {}".format(
                        iter + 1, max_iters, loss_hist[-1]
                    )
                )

        # End of epoch, compute the accuracies
        train_acc = 0
        val_acc = 0
        train_acc = compute_acc(model, data_train, labels_train)
        val_acc = compute_acc(model, data_val, labels_val)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        # Save the best parameters
        if val_acc > opt_val_acc:
            opt_val_acc = val_acc
            opt_params = model.net.params

        #training accuracies
        if verbose and epoch%10:
            print(
                "(Epoch {} / {}) Training Accuracy: {}, Validation Accuracy: {}".format(
                    epoch + 1, max_epochs, train_acc, val_acc
                )
            )

    return opt_params, loss_hist, train_acc_hist, val_acc_hist

features = df_normalized.columns[:-1]
target = "BEDS"

num_train = int(0.9 * len(df_normalized))
num_val = (len(df_normalized) - num_train)//2

small_data_dict = {
    "data_train": (np.array(df_normalized[features][:num_train]), np.array(df_normalized[target][:num_train])),
    "data_val": (np.array(df_normalized[features][num_train:]), np.array(df_normalized[target][num_train:])),
    "data_test": (np.array(test_normalized[features]), None)
}
model_sgd      = FullyConnectedNetwork()
loss_f_sgd     = cross_entropy()
# optimizer_sgd  = SGD(model_sgd.net, 1e-2)
optimizer_sgd  = Adam(model_sgd.net, lr=1e-3, beta1=0.95, beta2=0.999, t=0, eps=1e-8)

results_sgd = train_net(small_data_dict, model_sgd, loss_f_sgd, optimizer_sgd, batch_size=32,
                        max_epochs=75, show_every=100, verbose=True)

opt_params, loss_hist, train_acc_hist, val_acc_hist = results_sgd

loaded_model = FullyConnectedNetwork()
loaded_model.net.load(opt_params)
test_data, _ = small_data_dict["data_test"]

output = model_sgd.forward(test_data, False)
scores = softmax(output)
pred = np.argmax(scores, axis = 1)
predictions = pd.DataFrame({"BEDS": pred}).to_csv("output.csv", index = False)
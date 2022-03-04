from models.classifier import BinaryClassifier
import numpy as np
import torch
from tqdm.auto import trange
from models.flow import Flow



def label_classifier(data_loader_train, data_test, labels_test, n_steps = int(1e4), lr=1e-4, weight_decay=1e-4, activation="ReLU"):
    
    #? is data_dim and data_loader redundant? I think so

    data_dim = data_test.shape[-1]

    classifier = BinaryClassifier(data_dim, activation=activation)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)  

    classifier.train()
    classifier_loss = []
    print("Training Binary Label Classifier...")
    for n_step in trange(n_steps):
        data, labels = next(iter(data_loader_train))

        #TODO: add validation set
        
        optimizer.zero_grad()

        loss = classifier.loss(data, labels).mean()
        classifier_loss.append(loss.item())

        loss.backward()
        optimizer.step()
    classifier.eval();

    print("Binary Classifier Accuracy: ", ((classifier.forward(data_test) > 0.0) == labels_test).float().mean().item())


def get_probability_flow(data_loader_train, data_test, context_test, n_steps = int(1e4), lr=1e-4, weight_decay=1e-4):

    data_dim = data_test.shape[-1]
    context_dim = context_test.shape[-1]

    probability_flow = Flow(data_dim=data_dim, 
                context_dim=context_dim, 
                n_layers = 3,
                transform_type = 'MaskedPiecewiseQuadraticAutoregressiveTransform', 
                num_bins = 50,
                tails='linear',
                tail_bound=6.0,
    )
    optimizer = torch.optim.AdamW(probability_flow.parameters(), lr=1e-4, weight_decay=1e-4)

    probability_loss = []

    probability_flow.train()
    for n_step in trange(n_steps):
        data, context = next(iter(data_loader_train)) #! Data loader should have both context and labels. 
        context=context.unsqueeze(1)

        optimizer.zero_grad()

        loss = -probability_flow.log_prob(inputs=data, context=context).mean()
        probability_loss.append(loss.item())

        loss.backward()
        optimizer.step()
    probability_flow.eval();
    return probability_flow
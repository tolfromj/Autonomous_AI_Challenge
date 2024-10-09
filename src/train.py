from .models import HuggingfaceModel



def train():
    model = HuggingfaceModel()
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    for epoch in num_epochs:
        for batch in train_loader:
            images, targets = batch
            # images = tuple(torch.tensor[C, H, W], torch.tensor[C, H, W], ...)
            # images = [torch.tensor(image).cuda() for image in images]
            # torchmodels model 사용을 할때 list
            images = torch.stack(images).cuda()
            outputs = model(images, targets)

            loss = outputs["loss"]
            predicted_boxes = outputs["predicted_boxes"]

            loss.backward()
            optimizer.step()

            break
        break


import sys

sys.exit(0)


            # loss = model(images, targets)
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            # scheduler.step()
            # print(f"Epoch {epoch} Loss: {loss.item()}")

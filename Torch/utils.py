
def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255. # dataset이라는 클래스 받아오고 텐서로 바꿔놓음. 값이 int로 되있어가지구 float형으로 바꿔줘야함 .
    y = dataset.targets #  target에는 int형으로 들어가있음, 물론 텐서로 ,.

    if flatten:
        x = x.view(x.size(0), -1) # view 쓰면 60000, 28 ,28 에서 60000, 784 로 맞춰줌.

    return x, y

import tensorflow as tf
import matplotlib.pyplot as plt
keras=tf.keras
layers=keras.layers
def generator_model():
    model=keras.Sequential()#
    model.add(layers.Dense(256,input_shape=(100,),use_bias=False))#输入形状100是我输入噪声的形状,生成器一般都不使用BIAS
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())#GAN中一般使用LeakyRelu函数来激活
    model.add(layers.Dense(512,use_bias=False))#生成器一般都不使用BIAS
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(28*28*1,use_bias=False,activation='tanh'))#生成能够调整成我们想要的图片形状的向量
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((28,28,1)))#这里进行修改向量的形状，可以直接使用layers的reshape
    return model
def discriminator_model():
    model=keras.Sequential()
    model.add(layers.Flatten())#图片是一个三维数据，要输入到全连接层之前，先使用flatten层压平为一维的
    model.add(layers.Dense(512,use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(256,use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1))#最后输出为0,1只需要一层
    return model
(x_train,y_train),_=keras.datasets.mnist.load_data()
x_train=tf.expand_dims(x_train,axis=-1)#这里由于输入的手写体是只有两个维度的，所以这里我扩展最后一个维度
x_train=tf.cast(x_train,tf.float32)
x_train=x_train/255.0
x_train=x_train*2-1#将图片数据规范到[-1,1]
BATCH_SIZE=256
BUFFER_SIZE=60000#每次训练弄乱的大小
dataset=tf.data.Dataset.from_tensor_slices(x_train)
dataset=dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
loss_object=keras.losses.BinaryCrossentropy(from_logits=True)#损失这里使用二分类交叉熵损失，没有激活是logits
def discriminator_loss(real_out,fake_out):
    real_loss=loss_object(tf.ones_like(real_out),  real_out)
    fake_loss=loss_object(tf.zeros_like(fake_out),fake_out)
    return real_loss+fake_loss
#这里判别器使用的损失是计算我们人为制造的0,1标签与判别器模型输出的做计算，最终返回二者相加
def generator_loss(fake_out):
    fake_loss=loss_object(tf.ones_like(fake_out),fake_out)
    return fake_loss
#生成器计算损失当然是希望判别器都把他当真，所以是与1做计算

generator_opt=keras.optimizers.Adam(1e-4)
discriminator_opt=keras.optimizers.Adam(1e-4)#定义两个模型的优化器
EPOCHS = 10
noise_dim = 100  # 输入噪声的维度
num = 16  # 每次随机绘画16张图
seed = tf.random.normal(shape=([num, noise_dim]))  # 制作用于生成图片的向量
gen_model = generator_model()
dis_model = discriminator_model()


# 初始化这两个模型
# 定义训练步骤
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gentape, tf.GradientTape() as disctape:
        real_output = dis_model(images, training=True)
        fake_image = gen_model(noise, training=True)
        fake_output = dis_model(fake_image, training=True)
        gen_loss = generator_loss(fake_output)
        dis_loss = discriminator_loss(real_output, fake_output)
    grad_gen = gentape.gradient(gen_loss, gen_model.trainable_variables)
    grad_dis = disctape.gradient(dis_loss, dis_model.trainable_variables)
    generator_opt.apply_gradients(zip(grad_gen, gen_model.trainable_variables))
    discriminator_opt.apply_gradients(zip(grad_dis, dis_model.trainable_variables))


# 在每次训练后绘图
def generate_plot_img(gen_model, test_noise):
    pre_img = gen_model(test_noise, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(pre_img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((pre_img[i, :, :, 0] + 1) / 2, cmap='gray')
        # 这里cmap限定绘图的颜色空间，灰度图
        plt.axis('off')
    plt.show()  # 将16张图片一起显示出来

def train(dataset, epochs):
    for epoch in range(epochs):
        for img in dataset:
            train_step(img)
            print('-',end='')
        generate_plot_img(gen_model,seed)#绘制图片
train(dataset,EPOCHS)#这里EPOCHS我设置为100

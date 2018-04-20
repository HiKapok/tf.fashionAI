
# import requests

# def download_file_from_google_drive(id, destination):
#     def get_confirm_token(response):
#         for key, value in response.cookies.items():
#             if key.startswith('download_warning'):
#                 return value

#         return None

#     def save_response_content(response, destination):
#         CHUNK_SIZE = 32768

#         with open(destination, "wb") as f:
#             for chunk in response.iter_content(CHUNK_SIZE):
#                 if chunk: # filter out keep-alive new chunks
#                     f.write(chunk)

#     URL = "https://docs.google.com/uc?export=download"

#     session = requests.Session()

#     response = session.get(URL, params = { 'id' : id }, stream = True)
#     token = get_confirm_token(response)

#     if token:
#         params = { 'id' : id, 'confirm' : token }
#         response = session.get(URL, params = params, stream = True)

#     save_response_content(response, destination)

# #?id=
# if __name__ == "__main__":

#     # TAKE ID FROM SHAREABLE LINK
#     file_id = '1AwG0nWFUrikd17xQpTmAj2LcwK-MbNqJ'
#     # DESTINATION FILE ON YOUR DISK
#     destination = './dd.txt'
#     download_file_from_google_drive(file_id, destination)


import tensorflow as tf

targets = tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                       [[[11, 12], [13, 14]], [[15, 16], [17, 18]]]])

pred_outputs = tf.zeros_like(targets)

sss = tf.losses.mean_squared_error(targets, pred_outputs, weights=1.0,
                                    loss_collection=None,
                                    reduction=tf.losses.Reduction.NONE)
num_topk = 1
sss = tf.reduce_mean(tf.reshape(sss, [2, 2, -1]), axis=-1)
gather_col = tf.nn.top_k(sss, k=num_topk, sorted=True)[1]

gather_row = tf.reshape(tf.tile(tf.reshape(tf.range(2), [-1, 1]), [1, num_topk]), [-1, 1])

gather_indcies = tf.stack([gather_row, gather_col], axis=-1)

select_heatmap = tf.gather_nd(targets, gather_indcies)

sess = tf.Session()
table = tf.contrib.lookup.HashTable(
    tf.contrib.lookup.KeyValueTensorInitializer(tf.constant([0,1,2], dtype=tf.int64), tf.constant([1,2,-1], dtype=tf.int64)), 0)
out = table.lookup(tf.constant([0,1,2,3,4], dtype=tf.int64))
sess.run(tf.group([tf.local_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()]))
with sess.as_default():
    #table.init.run()
    print(sss.eval())
    print(gather_col.eval())
    print(gather_row.eval())
    print(gather_indcies.eval())
    print(select_heatmap.eval())



heatmap_sigma = 1.
ksize = 6 * heatmap_sigma + 1.

x = tf.expand_dims(tf.range(tf.round(ksize), delta=1, dtype=tf.float32), axis=1)
y = tf.transpose(x, [1, 0])
hhh = tf.exp(- ((x - ksize/2.) ** 2 + (y - ksize/2.) ** 2) / (2 * heatmap_sigma ** 2))



hhh_filter = tf.reshape(hhh, [tf.round(ksize), tf.round(ksize), 1, 1])
hhh_filter = tf.transpose(hhh_filter, [1, 0, 2, 3])


filtered_x = tf.nn.conv2d(image_resized, sobel_x_filter,
                          strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None)







pred = tf.constant([[[[1.11,1.12],[1.13,1.14]], [[1.21,1.22],[1.23,1.24]]], [[[2.11,2.12],[2.13,2.14]], [[2.21,2.22],[2.23,2.24]]]])
pred = tf.segment_mean(pred, [0,0])

im = tf.constant([[[[1.11,1.12,1.],[1.13,1.14,1.]], [[1.21,1.22,1.],[1.23,1.24,1.]]], [[[2.11,2.12,1.],[2.13,2.14,1.]], [[2.21,2.22,1.],[2.23,2.24,1.]]]])
#pred = tf.reshape(tf.stack([im, im], axis=1), [-1, 2, 2, 3])
aaaaaaa1 = tf.tile(tf.reshape(tf.range(tf.shape(pred)[0]), [-1, 1]), [1, 2])
aaaaaaa2=tf.stack([aaaaaaa1,tf.constant([[1,0],[0,1]])], axis=-1)
indcesss = tf.constant([[[0,1],[0,0]],[[1,1],[1,0]]])

aaaaaaa = tf.gather_nd(pred, aaaaaaa2)

d = {'d':12, 'f':0.12344, 'c':555.5767}
items = d.items()
print(', '.join(['%s=%.6f' % (k, v) for k,v in items]))

heatmap_size=64
pred_heatmap = tf.one_hot([34*heatmap_size+23, 1*heatmap_size+60, 32*heatmap_size+1], heatmap_size*heatmap_size, on_value=1., off_value=0., axis=-1, dtype=tf.float32)
pred_max = tf.reduce_max(pred_heatmap, axis=-1)
pred_indices = tf.argmax(pred_heatmap, axis=-1)
pred_x, pred_y = tf.cast(tf.floormod(pred_indices, heatmap_size), tf.float32), tf.cast(tf.floordiv(pred_indices, heatmap_size), tf.float32)


a = tf.losses.mean_squared_error(tf.constant([[0,1],[1,2],[2,3],[3,4],[4,5]], dtype=tf.int64),tf.constant([[1,2],[2,3],[3,4],[4,5],[5,6]], dtype=tf.int64), weights=1.0/5., loss_collection=None, reduction=tf.losses.Reduction.MEAN)


aa = tf.reduce_sum(tf.squared_difference(tf.constant([[0,1],[1,2],[2,3],[3,4],[4,5]], dtype=tf.int64),tf.constant([[1,2],[2,3],[3,4],[4,5],[5,6]], dtype=tf.int64)), axis=-1)
b = tf.metrics.mean_absolute_error(aa, tf.zeros_like(aa))
#tf.metrics.mean_squared_error(,
                                #weights=1.0*2,
                                #name='last_pred_mse')
sess = tf.Session()
table = tf.contrib.lookup.HashTable(
    tf.contrib.lookup.KeyValueTensorInitializer(tf.constant([0,1,2], dtype=tf.int64), tf.constant([1,2,-1], dtype=tf.int64)), 0)
out = table.lookup(tf.constant([0,1,2,3,4], dtype=tf.int64))
sess.run(tf.group([tf.local_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()]))
with sess.as_default():
    #table.init.run()
    print(pred.eval())
    print(aaaaaaa1.eval())
    print(aaaaaaa2.eval())
    #print(b[0].eval())
    # print(a.eval())
    # print(b[1].eval())
    # print(pred_x.eval())

    # print(pred_y.eval())


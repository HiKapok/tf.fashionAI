import tensorflow as tf


d = {'d':12, 'f':0.12344, 'c':555.5767}
items = d.items()
print(', '.join(['%s=%.6f' % (k, v) for k,v in items]))

sess = tf.Session()
table = tf.contrib.lookup.HashTable(
    tf.contrib.lookup.KeyValueTensorInitializer(tf.constant([0,1,2], dtype=tf.int64), tf.constant([1,2,-1], dtype=tf.int64)), 0)
out = table.lookup(tf.constant([0,1,2,3,4], dtype=tf.int64))
with sess.as_default():
    table.init.run()
    print(out.eval())

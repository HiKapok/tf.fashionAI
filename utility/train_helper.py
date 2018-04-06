import os

import tensorflow as tf

def get_init_fn_for_scaffold(flags):
    flags_checkpoint_path = flags.checkpoint_path
    # Warn the user if a checkpoint exists in the model_dir. Then ignore.
    if tf.train.latest_checkpoint(flags.model_dir):
        tf.logging.info('Ignoring --checkpoint_path because a checkpoint already exists in %s' % flags.model_dir)
        return None
    if flags_checkpoint_path is None:
        return None
    exclusions = []
    if flags.checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in flags.checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        excluded = False
        #print(var.op.name)
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    if flags.checkpoint_model_scope is not None:
        if flags.checkpoint_model_scope.strip() == '':
            variables_to_restore = {var.op.name.replace(flags.model_scope + '/', flags.checkpoint_model_scope): var for var in variables_to_restore}
        else:
            variables_to_restore = {var.op.name.replace(flags.model_scope, flags.checkpoint_model_scope): var for var in variables_to_restore}

    if tf.gfile.IsDirectory(flags_checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(flags_checkpoint_path)
    else:
        checkpoint_path = flags_checkpoint_path

    tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s' % (checkpoint_path, flags.ignore_missing_vars))

    # For DEBUG #
    # reader = tf.train.NewCheckpointReader(checkpoint_path)
    # variables = reader.get_variable_to_shape_map()
    # print('######################## variables in checkpoint ########################')
    # for ele in variables:
    #     print(ele)
    # print('######################### variables_to_restore #########################')
    # for ele in variables_to_restore:
    #     print(ele)
    # For DEBUG #
    if not variables_to_restore:
        raise ValueError('variables_to_restore cannot be empty')
    if flags.ignore_missing_vars:
        reader = tf.train.NewCheckpointReader(checkpoint_path)
        if isinstance(variables_to_restore, dict):
            var_dict = variables_to_restore
        else:
            var_dict = {var.op.name: var for var in variables_to_restore}
        available_vars = {}
        for var in var_dict:
            if reader.has_tensor(var):
                available_vars[var] = var_dict[var]
            else:
                tf.logging.warning('Variable %s missing in checkpoint %s', var, checkpoint_path)
        variables_to_restore = available_vars
    if variables_to_restore:
        saver = tf.train.Saver(variables_to_restore, reshape=False)
        saver.build()
        def callback(scaffold, session):
            saver.restore(session, checkpoint_path)
        return callback
    else:
        tf.logging.warning('No Variables to restore')
        return None

def get_latest_checkpoint_for_evaluate(flags):
    flags_checkpoint_path = flags.checkpoint_path
    # Warn the user if a checkpoint exists in the model_dir. Then ignore.
    if tf.train.latest_checkpoint(flags.model_dir):
        tf.logging.info('Ignoring --checkpoint_path because a checkpoint already exists in %s' % flags.model_dir)
        return None

    if tf.gfile.IsDirectory(flags_checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(flags_checkpoint_path)
    else:
        checkpoint_path = flags_checkpoint_path

    tf.logging.info('Restore from %s.' % (checkpoint_path))

    return checkpoint_path


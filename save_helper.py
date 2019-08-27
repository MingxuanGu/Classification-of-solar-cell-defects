import tensorflow as tf
import os
import zipfile

# taken from TF source code since this is not available in TF 1.4
def simple_save(session, export_dir, inputs, outputs, legacy_init_op=None):
  signature_def_map = {
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          tf.saved_model.signature_def_utils.predict_signature_def(inputs, outputs)
  }
  print("build saveModel")
  b = tf.saved_model.builder.SavedModelBuilder(export_dir)
  print("add variables")
  b.add_meta_graph_and_variables(
      session,
      tags=[tf.saved_model.tag_constants.SERVING],
      signature_def_map=signature_def_map,
      assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
      legacy_init_op=legacy_init_op,
      clear_devices=True)
  print("save")
  b.save()


def zip_dir(export_dir, target_file):
    paths = [p for p in export_dir.glob('**/*')]
    print(paths)

    zipf = zipfile.ZipFile(str(target_file), 'w', zipfile.ZIP_DEFLATED)

    for path in paths:
        zipf.write(str(path), str(path.relative_to(export_dir)))

    zipf.close()



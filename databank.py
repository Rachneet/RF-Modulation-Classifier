import sqlite3
import h5py as h5
import numpy as np
import io

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

path = "/home/rachneet/rf_dataset_inets/vsg_no_intf_all_normed.h5"
file = h5.File(path,'r')
iq, labels, snrs = file['iq'], file['labels'], file['snrs']
# print(iq[0])

conn = sqlite3.connect("modulation.db", detect_types=sqlite3.PARSE_DECLTYPES)
cur = conn.cursor()   # allows executing sql queries
# cur.execute("""DROP TABLE raw_iq;""")
cur.execute("""CREATE TABLE IF NOT EXISTS raw_iq(
                iq ARRAY,
                label ARRAY,
                snr INT);
""")
conn.commit()
# cur.execute("""INSERT INTO raw_iq(iq, label, snr)
#    VALUES(?, ?, ?)""",(iq[0], labels[0], int(snrs[0])))
# conn.commit()

cur.execute("SELECT * FROM raw_iq;")
one_result = cur.fetchone()
print(one_result)
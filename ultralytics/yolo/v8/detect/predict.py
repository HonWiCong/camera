from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.engine.predictor import BasePredictor
import hydra
import torch
import easyocr
import time
import mysql.connector
import re
import json

# python -m pip install awsiotsdk
from awscrt import io, mqtt, auth, http
from awsiot import mqtt_connection_builder

localDatabase = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="smart_parking"
)

endpoint = "a27eliy2xg4c5e-ats.iot.us-east-1.amazonaws.com"
cert_filepath = r"D:\Users\User\Programming\Swinburne Project\IOT\Hardware Program\mqtt\44bdbb017ed61e3180473d7562a7219625694010abfe0315ab96632a7fe8402b-certificate.pem.crt"
pri_key_filepath = r"D:\Users\User\Programming\Swinburne Project\IOT\Hardware Program\mqtt\44bdbb017ed61e3180473d7562a7219625694010abfe0315ab96632a7fe8402b-private.pem.key"
ca_filepath = r"D:\Users\User\Programming\Swinburne Project\IOT\Hardware Program\mqtt\AmazonRootCA1.pem"
client_id = "rpi_carplate_detector"

# RPi
# cert_filepath = r"/home/pi/RPi/mqtt/44bdbb017ed61e3180473d7562a7219625694010abfe0315ab96632a7fe8402b-certificate.pem.crt"
# pri_key_filepath = r"/home/pi/RPi/mqtt/44bdbb017ed61e3180473d7562a7219625694010abfe0315ab96632a7fe8402b-private.pem.key"
# ca_filepath = r"/home/pi/RPi/mqtt/AmazonRootCA1.pem"

event_loop_group = io.EventLoopGroup(1)
host_resolver = io.DefaultHostResolver(event_loop_group)
client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)

io.init_logging(getattr(io.LogLevel.NoLogs, 'NoLogs'), 'stderr')

mqtt_connection = mqtt_connection_builder.mtls_from_path(
    endpoint=endpoint,
    cert_filepath=cert_filepath,
    pri_key_filepath=pri_key_filepath,
    client_bootstrap=client_bootstrap,
    ca_filepath=ca_filepath,
    client_id=client_id,
    clean_session=False,
    keep_alive_secs=6
)

print("Connecting to {} with client ID '{}'...".format(endpoint, client_id))
connected_future = mqtt_connection.connect()
connected_future.result()
if connected_future.done():
    print("Connected!")
else:
    print("Connection failed")

class DetectionPredictor(BasePredictor):
    def is_valid_plate_number(self, plate_number):
        pattern = r'^[A-Za-z]{2,3}\d{1,4}[A-Za-z]?$'
        return bool(re.match(pattern, plate_number))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_images = {}
        self.processing_lock = False
        self.last_process_time = 0
        self.process_interval = 0.5  # 2 FPS (1 second / 2 = 0.5 seconds)
        self.last_frame_time = 0
        self.frame_interval = 0.5  # 2 FPS (1 second / 2 = 0.5 seconds)
        self.isEntering = True

    def extract_plate_number(self, img):
        cursor = localDatabase.cursor(dictionary=True)

        # 1 represents processing
        cursor.execute("UPDATE variables SET value = 1 WHERE name = 'is_processing_carplate'")
        localDatabase.commit()

        reader = easyocr.Reader(['en'])
        result = reader.readtext(img)
        
        if result:
            plate_number = result[0][1]
            # 2 represents completed
            cursor.execute("UPDATE variables SET value = 2 WHERE name = 'is_processing_carplate'")
            localDatabase.commit()
            return plate_number
        else:
            # 3 represents failed or error
            cursor.execute("UPDATE variables SET value = 3 WHERE name = 'is_processing_carplate'")
            localDatabase.commit()
            return None

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(
                img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()

        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + \
            ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string

        # Normalize the bounding box coordinates
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        # Process the detections
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                        gn).view(-1).tolist()  # normalized xywh
                # label format
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                self.annotator.box_label(xyxy, label, color=colors(c, True))

            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy, imc, file=self.save_dir / 'crops' /
                             self.model.model.names[c] /
                             f'{self.data_path.stem}.jpg', BGR=True)

            xyxy_tuple = tuple(xyxy)
            if xyxy_tuple not in self.cached_images:
                self.cached_images[xyxy_tuple] = im0[int(xyxy[1]):int(xyxy[3]),
                                                     int(xyxy[0]):int(xyxy[2])].copy()

            if not self.processing_lock:
                self.processing_lock = True
                current_time = time.time()
                if current_time - self.last_process_time >= self.process_interval:
                    self.last_process_time = current_time
                    plate_number = self.extract_plate_number(self.cached_images[xyxy_tuple])
                    
                    # remove all the empty spaces from the plate number
                    if plate_number is not None: plate_number = plate_number.replace(" ", "")
                    if plate_number is not None and self.is_valid_plate_number(plate_number):
                        cursor = localDatabase.cursor(dictionary=True)
                        cursor.execute("SELECT * FROM variables WHERE name = 'is_entering'")
                        self.isEntering = cursor.fetchone()["value"]

                        # if the car is entering
                        if self.isEntering:
                            # check if the car is already in the database and the leave_at is null
                            cursor.execute(f"SELECT * FROM car_entry_exit_log WHERE carplate = '{plate_number}' AND exit_at IS NULL ORDER BY enter_at DESC LIMIT 1")
                            record = cursor.fetchone()
                            print("Record,", record)

                            if record is None:
                                sql = {
                                    sql: f"INSERT INTO car_entry_exit_log (carplate, enter_at) VALUES ('{plate_number}', NOW())"
                                }
                                mqtt_connection.publish(topic="rpi/carplate_post_request", payload=json.dumps(sql), qos=2)
                                cursor = localDatabase.cursor(dictionary=True)
                                cursor.execute(f"INSERT INTO car_entry_exit_log (carplate, enter_at) VALUES ('{plate_number}', NOW())")
                                localDatabase.commit()
                            else:
                                print("Car is already in the database, you can't enter again.")
                        
                        # if the car is exiting
                        elif not self.isEntering:
                            cursor.execute(f"SELECT * FROM car_entry_exit_log WHERE carplate = '{plate_number}' ORDER BY enter_at DESC LIMIT 1")
                            record = cursor.fetchone()
                            if record is None:
                                print("Something went wrong")
                            else:
                                sql = {
                                    sql: f"UPDATE car_entry_exit_log SET exit_at = NOW(), duration = TIMESTAMPDIFF(MINUTE, enter_at, NOW()) WHERE carplate = '{plate_number}' ORDER BY enter_at DESC LIMIT 1"
                                }
                                mqtt_connection.publish(topic="rpi/carplate_post_request", payload=json.dumps(sql), qos=2)
                                cursor = localDatabase.cursor(dictionary=True)
                                cursor.execute(f"""
                                    UPDATE car_entry_exit_log 
                                    SET exit_at = NOW(), 
                                        duration = TIMESTAMPDIFF(MINUTE, enter_at, NOW())
                                    WHERE carplate = '{plate_number}' 
                                    ORDER BY enter_at DESC 
                                    LIMIT 1
                                """)
                                localDatabase.commit()

                self.processing_lock = False

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()

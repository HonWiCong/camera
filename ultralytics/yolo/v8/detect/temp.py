if not self.processing_lock:
	self.processing_lock = True
	current_time = time.time()
	if current_time - self.last_process_time >= self.process_interval:
		self.last_process_time = current_time
		plate_number = self.extract_plate_number(
			self.cached_images[xyxy_tuple])
		# remove all the empty spaces from the plate number
		if plate_number:
			plate_number = plate_number.replace(" ", "")
		if plate_number and self.is_valid_plate_number(plate_number):
			cursor = database.cursor(dictionary=True)
			cursor.execute("SELECT * FROM variables WHERE name='is_entering'")
			result = cursor.fetchone()
			self.isEntering = result["value"]

			# enter
			if self.isEntering:
				cursor.execute(f"SELECT * FROM car_entry_exit_log WHERE carplate = '{plate_number}' ORDER BY enter_at DESC LIMIT 1")
				result = cursor.fetchone()
				if result:
					last_enter_time = result["enter_at"]
					current_time = time.time()
					print("Current Time: ", current_time)
					print("Last Enter Time: ", last_enter_time)
					if last_enter_time:
						last_enter_timestamp = time.mktime(
							last_enter_time.timetuple())
						time_difference = current_time - last_enter_timestamp
						print("Time Difference: ", time_difference)
						if time_difference < 15:
							self.processing_lock = False
							return log_string
						cursor = database.cursor(dictionary=True)
						cursor.execute(
							f"INSERT INTO car_entry_exit_log (carplate, enter_at) VALUES ('{plate_number}', NOW())")
						database.commit()

			# exit
			elif not self.isEntering:
				cursor.execute(
					f"SELECT * FROM car_entry_exit_log WHERE carplate = '{plate_number}' ORDER BY leave_at DESC LIMIT 1")
				result = cursor.fetchone()
				if result:
					last_exit_time = result["leave_at"]
					current_time = time.time()
					print("Current Time: ", current_time)
					print("Last Exit Time: ", last_exit_time)
					if last_exit_time:
						last_exit_timestamp = time.mktime(
							last_exit_time.timetuple())
						time_difference = current_time - last_exit_timestamp
						print("Time Difference: ", time_difference)
						if time_difference < 15:
							self.processing_lock = False
							return log_string
						cursor = database.cursor(dictionary=True)
						cursor.execute(
							f"SELECT * FROM car_entry_exit_log WHERE carplate = '{plate_number}' AND leave_at IS NULL ORDER BY enter_at DESC LIMIT 1")
						result = cursor.fetchone()
						if result:
							cursor.execute(
								f"UPDATE car_entry_exit_log SET leave_at = NOW() WHERE id = {result['id']}")
							# then also calculate the duration of the car in the parking lot in minutes
							enter_at = result["enter_at"]
							leave_at = time.time()
							time_difference = leave_at - \
								time.mktime(enter_at.timetuple())
							duration = time_difference / 60
							cursor.execute(
								f"UPDATE car_entry_exit_log SET duration = {duration} WHERE id = {result['id']}")
							database.commit()
						else:
							print("Something strange happened")

	self.processing_lock = False
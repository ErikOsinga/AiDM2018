for seed in 123 124 125 126 127; 
do 
	for sign_t in 0.45 0.5 0.55; 
	do 
		time python assignment3_testing_values.py $seed ./user_movie.npy 64 16 $sign_t
		time python assignment3_testing_values.py $seed ./user_movie.npy 90 30 $sign_t
		time python assignment3_testing_values.py $seed ./user_movie.npy 92 23 $sign_t
		time python assignment3_testing_values.py $seed ./user_movie.npy 100 10 $sign_t
		time python assignment3_testing_values.py $seed ./user_movie.npy 150 30 $sign_t
		time python assignment3_testing_values.py $seed ./user_movie.npy 150 50 $sign_t
	done
done


for f in $(ls | grep moves); do
    if [ -s "$f" ] 
    then
	echo ""
	#echo "$f has some data."
        # do something as file has data
    else
	num=$(echo "$f" | cut -d '_' -f 1)
	echo $num
	rm "${num}_details.txt"
	rm "${num}_moves.txt"
	rm "${num}_tags.txt"
	rm "${num}.pgn"
        # do something as file is empty 
    fi
done

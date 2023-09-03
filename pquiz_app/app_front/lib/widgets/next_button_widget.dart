import 'package:flutter/material.dart';

class NextButton extends StatelessWidget {
  const NextButton({Key? key, required this.pressed}) : super(key: key);

  final bool pressed;

  @override
  Widget build(BuildContext context) {

    Color color =  Color(0xFF2F3542);

    if(pressed){ //정답인지를 확인한 후 다음 문제로 넘어갈 시
      color = Color(0xFF2F3542);
      return Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(20),
          color: color,
        ),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 10),
          child: Text(
            'Next Question',
            style: TextStyle(
              fontSize: 16.0,
              color: Colors.white,
            ),
          ),
        ),
      );
    } else{ //처음에 문제를 풀고 버튼을 눌렀을 시
      return Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(20),
          color: color,
        ),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 10),
          child: Text(
            'Submission',
            style: TextStyle(
              fontSize: 16.0,
              color: Colors.white,
            ),
          ),
        ),
      );
    }
  }
}

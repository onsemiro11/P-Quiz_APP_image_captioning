// import 'package:commerce_quiz_qpp/constants/constants.dart';
import 'package:flutter/material.dart';

class QuestionWidget extends StatelessWidget {
  const QuestionWidget(
      {Key? key,
      required this.question,
      required this.indexAction,
      required this.totalQuestions})
      : super(key: key);

  final String question;
  final int indexAction;
  final int totalQuestions;

  @override
  Widget build(BuildContext context) {
    return Container(
      alignment: Alignment.centerLeft,
      padding: EdgeInsets.all(10.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.start,
            children: [
              Text(
                'Question ${indexAction + 1}/$totalQuestions',
                style: TextStyle(
                  fontSize: 18.0,
                  color: Colors.black,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          SizedBox(
            height: 10.0,
          ),
          Text(
            'What is the correct image description?',
            style: TextStyle(
              fontSize: 14.5,
              color: Colors.black,
            ),
          ),
          SizedBox(
            height: 25.0,
          ),
          Image.network('$question',
              width: 400, // 이미지의 너비
              height: 200, // 이미지의 높이
              fit: BoxFit.cover), // 이미지의 크기와 위젯의 크기를 맞춤)
        ],
      ),
    );
  }
}

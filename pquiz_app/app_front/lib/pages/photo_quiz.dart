import 'dart:io';

import 'package:animation_wrappers/animation_wrappers.dart';
import 'package:flutter/material.dart';
import 'package:flutter_countdown_timer/countdown_timer_controller.dart';
import 'package:flutter_countdown_timer/flutter_countdown_timer.dart';
import 'package:install_test/model/photo_question_model.dart';
import 'package:install_test/pages/result_photo_page.dart';
import 'package:install_test/widgets/next_button_widget.dart';
import 'package:install_test/widgets/options_widget.dart';
import 'package:quickalert/quickalert.dart';

import '../constants/constants.dart';
import '../model/photo_database.dart';
import '../model/question_model.dart';

class PhotoQuiz extends StatefulWidget {
  const PhotoQuiz({Key? key, required this.ipath, required this.caption})
      : super(key: key);
  final String ipath;
  final String caption;
  @override
  State<PhotoQuiz> createState() => _PhotoQuizState();
}

class _PhotoQuizState extends State<PhotoQuiz> {
  var data;
  var db = PhotoDBconnect(); //photo database 불러옴
  late Future _questions;
  late CountdownTimerController controller;
  int endTime = DateTime.now().millisecondsSinceEpoch + 1000 * 30;
  bool timeFinished = false;

  Future<List<Question>> getData() async {
    return db.fetchQuestions();
  }

  List<photoQuestion> questions = [];
  @override
  void initState() {
    _questions = getData();

    controller = CountdownTimerController(endTime: endTime, onEnd: onEnd);

    _questions.then((data) {
      var extractedData = data;
      questions = [
        // 하나의 문제를 photoQuestion으로 생성(shuffle을 위해)
        photoQuestion(
          id: 1,
          title: widget.ipath,
          options: {
            extractedData[0].options.keys.toList()[0]: false,
            extractedData[0].options.keys.toList()[1]: false,
            extractedData[0].options.keys.toList()[2]: false,
            widget.caption: true,
          },
        ),
      ];
    });

    super.initState();
  }

  void onEnd() {
    print('onEnd');
    timeFinished = true;
    Navigator.pushReplacement(
        context,
        MaterialPageRoute(
            builder: (context) =>
                ResultPhotoScreen(total: score, quesLength: 1)));
  }

  void navi(int questionLength) {
    if (navi == true) {
      Navigator.pushReplacement(
          context,
          MaterialPageRoute(
              builder: (context) =>
                  ResultPhotoScreen(total: score, quesLength: questionLength)));
    }
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  int index = 0;
  bool isPressed = false;
  int score = 0;
  bool isAlreadySelected = false;
  String? selectedOption;

  void showNextQuestion(int questionLength) {
    if (index + 1 == questionLength) {
      if (isPressed == false) {
        //문제 정답 선택 안했을 시
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          duration: Duration(milliseconds: 800),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20.0)),
          content: Text('Please select a anwser', softWrap: false),
          behavior: SnackBarBehavior.floating,
          margin: EdgeInsets.symmetric(vertical: 90.0, horizontal: 90.0),
          backgroundColor: Color(0xFF2F3542),
        ));
      } else {
        setState(() {
          if (timeFinished == false) {
            Navigator.pushReplacement(
                context,
                MaterialPageRoute(
                    builder: (context) => ResultPhotoScreen(
                        total: score, quesLength: questionLength)));
          } else if (timeFinished == true) {
            Navigator.pushReplacement(
                context,
                MaterialPageRoute(
                    builder: (context) => ResultPhotoScreen(
                        total: score, quesLength: questionLength)));
          }
        });
      }
    } else {
      if (isPressed) {
        setState(() { // 다음 문제로 넘어감
          index++;
          isPressed = false;
          isAlreadySelected = false;
        });
      } else { //선택 안했을 시
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          duration: Duration(milliseconds: 800),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20.0)),
          content: Text('Please select a anwser', softWrap: false),
          behavior: SnackBarBehavior.floating,
          margin: EdgeInsets.symmetric(vertical: 90.0, horizontal: 90.0),
          backgroundColor: Color(0xFF2F3542),
        ));
      }
    }
  }

  void checkAndUpdate(bool value, int index) {
    if (isAlreadySelected) {
      return;
    } else {
      if (value == true) {
        score++;
        setState(() {
          isPressed = true;
          isAlreadySelected = true;
        });
      } else if (value == false) {
        setState(() {
          isPressed = true;
          isAlreadySelected = true;
        });
      }
    }
    if (value == true) {
      //정답일 시
      QuickAlert.show(
        context: context,
        customAsset: 'assets/images/correct.JPG',
        title: 'Correct Answer',
        titleColor: Color.fromARGB(255, 93, 156, 89),
        type: QuickAlertType.success,
        confirmBtnText: 'O K',
        confirmBtnColor: Color.fromARGB(255, 93, 156, 89),
      );
    } else if (value == false) { //오답일 시
      QuickAlert.show(
        context: context,
        customAsset: 'assets/images/incorrect.JPG',
        title: 'Wrong Answer',
        titleColor: Color.fromARGB(255, 223, 46, 56),
        type: QuickAlertType.success,
        confirmBtnText: 'O K',
        confirmBtnColor: Color.fromARGB(255, 223, 46, 56),
      );
    }
  }

  void startOver() {
    setState(() {
      index = 0;
      score = 0;
      isPressed = false;
      isAlreadySelected = false;
    });
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder(
      future: _questions as Future<List<Question>>,
      builder: (ctx, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return Center(
            child: CircularProgressIndicator(
                color: backColor),
          );
        }
        if (snapshot.connectionState == ConnectionState.done) {
          if (snapshot.hasError) {
            return Center(
              child: CircularProgressIndicator(
                color: backColor,
              ),
            );
          } else if (snapshot.hasData) {
            return Scaffold(
              body: FadeAnimation(
                duration: Duration(milliseconds: 1000),
                child: Container(
                  color: Color(0xFFF4E3E3),
                  child: SafeArea(
                    child: Column(
                      children: [
                        CountdownTimer(
                          controller: controller,
                          onEnd: onEnd,
                          endTime: endTime,
                        ),
                        Container(
                          width: double.infinity,
                          padding: EdgeInsets.all(15.0),
                          child: Container(
                            padding: EdgeInsets.all(5.0),
                            color: Colors.white,
                            height: 600.0,
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  mainAxisAlignment: MainAxisAlignment.start,
                                  children: [
                                    Text(
                                      'Your Image Question',
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
                                  ),
                                ),
                                SizedBox(
                                  height: 25.0,
                                ),
                                Image.file(File(widget.ipath),
                                    width: 400,
                                    height: 220,
                                    fit: BoxFit.cover), // 이미지의 크기와 위젯의 크기를 맞춤)
                                SizedBox(
                                  height: 30.0,
                                ),

                                for (int i = 0;
                                    i < questions[0].options.length;
                                    i++)
                                  FadeAnimation(
                                    child: GestureDetector(
                                      onTap: () => checkAndUpdate(
                                          questions[0]
                                                  .options
                                                  .values
                                                  .toList()[i] ==
                                              true,
                                          i 
                                          ),
                                      child: OptionCard(
                                        option: questions[0]
                                            .options
                                            .keys
                                            .toList()[i],
                                        pressed: isPressed ? true : false,
                                        correct: questions[0]
                                            .options
                                            .values
                                            .toList()[i],
                                      ),
                                    ),
                                  ),
                              ],
                            ),
                          ),
                        ),
                        SizedBox(
                          height: 20.0,
                        ),
                        GestureDetector(
                          onTap: () {
                            showNextQuestion(questions.length); //다음문제로 넘어감
                          },
                          child: Padding(
                            padding:
                                const EdgeInsets.symmetric(horizontal: 20.0),
                            child: NextButton(pressed: isPressed), //다음 버튼으로 이동 또는 선택 답안 제줄
                          ),
                        ),
                        SizedBox(
                          height: 20.0,
                        )
                      ],
                    ),
                  ),
                ),
              ),
              floatingActionButtonLocation:
                  FloatingActionButtonLocation.centerFloat,
              floatingActionButtonAnimator:
                  FloatingActionButtonAnimator.scaling,
            );
          } else {
            return Center(
              child: CircularProgressIndicator(color: backColor),
            );
          }
        }
        return Center(child: CircularProgressIndicator(color: backColor));
      },
    );
  }
}

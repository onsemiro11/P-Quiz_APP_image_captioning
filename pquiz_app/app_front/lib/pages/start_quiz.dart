import 'package:animation_wrappers/animation_wrappers.dart';
import 'package:flutter/material.dart';

import 'package:flutter_countdown_timer/countdown_timer_controller.dart';
import 'package:flutter_countdown_timer/flutter_countdown_timer.dart';
import 'package:install_test/constants/constants.dart';
import 'package:install_test/model/database.dart';
import 'package:install_test/model/question_model.dart';
import 'package:install_test/pages/result_page.dart';
import 'package:install_test/widgets/next_button_widget.dart';
import 'package:install_test/widgets/options_widget.dart';
import 'package:install_test/widgets/question_widget.dart';
import 'package:quickalert/quickalert.dart';


class StartQuiz extends StatefulWidget {
  const StartQuiz({Key? key}) : super(key: key);

  @override
  State<StartQuiz> createState() => _StartQuizState();
}

class _StartQuizState extends State<StartQuiz> {
  var db = DBconnect();
  late Future _questions;
  late CountdownTimerController controller;
  int endTime = DateTime.now().millisecondsSinceEpoch + 1000 * 300;
  bool timeFinished = false;

  Future<List<Question>> getData() async {
    return db.fetchQuestions();
  }

  @override
  void initState() {
    _questions = getData();
    super.initState();
    controller = CountdownTimerController(endTime: endTime, onEnd: onEnd);
  }

  void onEnd() {
    print('onEnd');
    timeFinished = true;
    Navigator.pushReplacement(
        context,
        MaterialPageRoute(
            builder: (context) => ResultScreen(total: score, quesLength: 4)));
  }

  void navi(int questionLength) {
    if (navi == true) {
      Navigator.pushReplacement(
          context,
          MaterialPageRoute(
              builder: (context) =>
                  ResultScreen(total: score, quesLength: questionLength)));
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
    if (index + 1 == questionLength) {  // 다 풀었을 시
      if (isPressed == false) { // option 선택 안했을 시
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          duration: Duration(milliseconds: 800),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20.0)),
          content: Text('Please select a anwser'),
          behavior: SnackBarBehavior.floating,
          margin: EdgeInsets.symmetric(vertical: 90.0, horizontal: 90.0),
          backgroundColor: Color(0xFF2F3542),
        ));
      } else {
        setState(() {
          if (timeFinished == false) {  //다풀고 시간 남음
            Navigator.pushReplacement(
                context,
                MaterialPageRoute(
                    builder: (context) => ResultScreen(
                        total: score, quesLength: questionLength)));
          } else if (timeFinished == true) { //다풀고 시간 초과
            Navigator.pushReplacement(
                context,
                MaterialPageRoute(
                    builder: (context) => ResultScreen(
                        total: score, quesLength: questionLength)));
          }
        });
      }
    } else {
      if (isPressed) {
        setState(() { //다음문제로 이동
          index++;
          isPressed = false;
          isAlreadySelected = false;
        });
      } else { //답 선택 안했을 시
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          duration: Duration(milliseconds: 800),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20.0)),
          content: Text('Please select a anwser'),
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
    if (value == true) { //정답일 시
      QuickAlert.show(
        context: context,
        customAsset: 'assets/images/correct.JPG',
        title: 'Correct Answer',
        titleColor: const Color.fromARGB(255, 93, 156, 89),
        type: QuickAlertType.success,
        confirmBtnText: 'Next Question',
        confirmBtnColor: Color.fromARGB(255, 93, 156, 89),
      );
    } else if (value == false) { // 오답일 시
      QuickAlert.show(
        context: context,
        customAsset: 'assets/images/incorrect.JPG',
        title: 'Wrong Answer',
        titleColor: const Color.fromARGB(255, 223, 46, 56),
        type: QuickAlertType.success,
        confirmBtnText: 'Next Question',
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
            var extractedData = snapshot.data as List<Question>;

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
                              children: [
                                QuestionWidget(
                                    question: extractedData[index].image_path,
                                    indexAction: index,
                                    totalQuestions:
                                        extractedData.length), //이부분이 문제 출력하는 부분
                                SizedBox(
                                  height: 10.0,
                                ),

                                for (int i = 0;
                                    i < extractedData[index].options.length;
                                    i++)
                                  FadeAnimation(
                                    child: GestureDetector(
                                      onTap: () => checkAndUpdate(
                                          extractedData[index]
                                                  .options
                                                  .values
                                                  .toList()[i] ==
                                              true,
                                          i
                                          ),
                                      child: OptionCard(
                                        option: extractedData[index]
                                            .options
                                            .keys
                                            .toList()[i],
                                        pressed: isPressed ? true : false,
                                        correct: extractedData[index]
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
                            showNextQuestion(extractedData.length);//다음문제로 넘어감
                          },
                          child: Padding(
                            padding:
                                const EdgeInsets.symmetric(horizontal: 20.0),
                            child: NextButton(pressed: isPressed),//다음 버튼으로 이동 또는 선택 답안 제줄
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

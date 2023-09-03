import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:install_test/Screen/Loading.dart';
import 'package:install_test/Screen/select_photo_options_screen.dart';
import 'package:install_test/pages/start_quiz.dart';

//첫 main 화면
class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  static const id = 'main_screen';

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  var data;

  Future _pickImage(ImageSource source) async {
    try {
      final image = await ImagePicker().pickImage(source: source);
      if (image == null) return;

      setState(() {
        Navigator.push(context, MaterialPageRoute(builder: (context) {
          return Loading(ipath: image.path);
        }));
      });
    } on PlatformException catch (e) {
      print(e);
    }
  }

  // 사진 선택 옵션 보여주기
  void _showSelectPhotoOptions(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true, // 스크롤 가능 여부
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(
          top: Radius.circular(25.0),
        ),
      ),
      builder: (context) => DraggableScrollableSheet(
          initialChildSize: 0.28, // 초기 상태의 드래그 가능한 시트가 차지하는 높이 비율
          maxChildSize: 0.4, // 사용자가 드래그하여 시트가 최대로 확장시 높이 비율
          minChildSize: 0.28, // 사용자가 드래그하여 시트가 최소로 축소시 높이 비율
          expand: false, // true일 경우 사용자가 모든 내용 볼 수 있게 시트 자동 확장
          builder: (context, scrollController) {
            // 드래그 가능한 시트 내용 구성
            return SingleChildScrollView(
              controller: scrollController,
              child: SelectPhotoOptionsScreen(
                // 사진 선택 옵션 보여줌
                onTap: _pickImage, // 터치(클릭)할 떄 호출되는 함수
              ),
            );
          }),
    );
  }
  //바로 문제 풀이 부분
  void _StartQuiz() {
    Navigator.push(context, MaterialPageRoute(builder: (context) {
      return StartQuiz();
    }));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFF4E3E3),
      body: SafeArea(
        // 디바이스 상단 노치나 하단 바 등과 겹치지 않도록 화면 내용 제한
        child: Padding(
          // 화면의 여백 지정하여 내용을 감싸고 있는 위젯
          padding:
              const EdgeInsets.only(left: 20, right: 20, bottom: 30, top: 20),
          // 세로 방향으로 위젯 나열
          child: Column(
            crossAxisAlignment:
                CrossAxisAlignment.center, // 가로 방향에서 위젯들이 어떻게 정렬될지 설정
            mainAxisAlignment:
                MainAxisAlignment.center, // 세로 방향에서 위젯들이 어떻게 정렬될지 설정
            children: [
              const Column(
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      SizedBox(
                        height: 30,
                      ),
                      Text(
                        'P-QUIZ',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: Color(0xFF2F3542),
                          fontSize: 41,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      SizedBox(height: 3),
                      Text(
                        "Let's make a quiz with your pictures!\nYou can also solve the quiz right away.",
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: Color(0xFF2F3542),
                          fontSize: 16,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
              const SizedBox(
                height: 42,
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Image.asset(
                    'assets/images/QUIZDOG.png',
                    height: 234,
                    width: 234,
                  ),
                  const SizedBox(
                    height: 40,
                  ),
                  ElevatedButton(
                    onPressed: () => _showSelectPhotoOptions(context),
                    style: ElevatedButton.styleFrom(
                      primary: Color(0xFF2F3542),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(17),
                      ),
                      minimumSize: const Size(10, 38),
                    ),
                    child: const Text(
                      "Quiz with pictures",
                      style: TextStyle(
                        fontFamily: 'Raleway',
                        fontWeight: FontWeight.w600,
                        fontSize: 16,
                        color: Colors.white,
                      ),
                    ),
                  ),
                  SizedBox(
                    height: 16,
                  ),
                  ElevatedButton(
                    onPressed: () => _StartQuiz(),
                    style: ElevatedButton.styleFrom(
                      primary: Color(0xFF2F3542),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(17),
                      ),
                      minimumSize: const Size(10, 38),
                      padding: EdgeInsets.symmetric(vertical: 10),
                    ),
                    child: const Text(
                      "Start quiz now",
                      style: TextStyle(
                        fontFamily: 'Raleway',
                        fontWeight: FontWeight.w600,
                        fontSize: 16,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

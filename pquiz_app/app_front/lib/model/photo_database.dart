import 'dart:convert';
import 'dart:math';
import 'package:install_test/model/question_model.dart';

import 'package:http/http.dart' as http;
//database.dart와 동일(다른 점 :1문제만 추출함)
class PhotoDBconnect {
  final url =
  Uri.parse('https://gist.githubusercontent.com/onsemiro11/58e56383b0123451ca6fe62a2fbdcadf/raw/f7c4d5c88ec75a0a4f9be34bc7835247faf71a0a/gistfile1.txt');

  Future<void> addQuestion(Question question) async {
    http.post(url,
        body: json.encode({'image_path': question.image_path, 'options': question.options}));
  }

  Future<List<Question>> fetchQuestions() async {
    final response = await http.get(url);
    if (response.statusCode != 200) {
      throw Exception('Failed to fetch questions');
    }

    final Map<String, dynamic>? data = json.decode(response.body);
    // print(data);
    if (data == null) {
      throw Exception('No data found');
    }

    final List<Question> newQuestions = [];
    data.forEach((key, value) {
      final newQuestion = Question(
        id: key,
        image_path: value['image_path'] ?? '',
        options: Map.castFrom(value['options'] ?? {}),
      );
      newQuestions.add(newQuestion);
    });

    List<Question> getRandomElements(List<Question> sourceList, int count) {
      final random = Random();
      List<Question> resultList = [];

      while (resultList.length < count) {
        int randomIndex = random.nextInt(sourceList.length);
        Question randomElement = sourceList[randomIndex];
        
        if (!resultList.contains(randomElement)) {
          resultList.add(randomElement);
        }
      }
      return resultList;
    }

    List<Question> extracted = getRandomElements(newQuestions, 1); //랜덤으로 한 문항 추출하여 option 활용 준비
    return extracted;
  }
}
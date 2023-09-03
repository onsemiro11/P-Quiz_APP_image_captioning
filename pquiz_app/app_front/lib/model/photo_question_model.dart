import 'dart:math';
// photoQuestion인 객체 생성 후 랜덤으로 option부분 shuffle 진행
class photoQuestion {
  final int id;
  final String title;
  final Map<String, bool> options;

  photoQuestion({
    required this.id,
    required this.title,
    required Map<String, bool> options,
  }) : options = Map.fromEntries(options.entries.toList()..shuffle(Random()));
}


